"""
Tarteel Live - Backend
-----------------------
FastAPI + WebSocket server that:
  1. Loads tarteel-ai/whisper-tiny-ar-quran ONCE at startup (not per-request).
  2. Accepts raw PCM16 audio chunks over a WebSocket.
  3. Runs simple energy-based VAD to detect speech vs silence.
  4. Transcribes speech chunks with Whisper and aligns the result against
     Surah Al-Fatiha word by word.
  5. Sends back per-word status (correct / wrong / pending) + a
     speaking/silence indicator.

Run with:
    uvicorn backend:app --host 0.0.0.0 --port 8000
(NO --reload in normal use — reload would reinitialize the model on every
file save. Use the /reload_model dev-only endpoint if you ever need to force
a reload without restarting the process.)
"""

import io
import os
import json
import re
import wave
import time
import difflib
import logging
from typing import List

# Once the model has been downloaded once, don't let transformers/huggingface_hub
# phone home on every startup just to check "is there a newer version?". That
# Hub-check is what was causing the repeated network calls / retries / crashes
# above (especially painful if the internet drops mid-check). Setting these
# BEFORE importing transformers forces it to use the local cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tarteel-live")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_ID = os.environ.get(
    "TARTEEL_MODEL_PATH",
    r"C:\Users\CHAND COMPUTER\Desktop\AudioSegment\models\whisper-100-112-600steps",
)
PROCESSOR_ID = "tarteel-ai/whisper-base-ar-quran"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Surah Al-Fatiha reference text (with basmalah), split into ayah -> words.
# Diacritics kept minimal / normalized so matching is more forgiving.
FATIHA_AYAT = [
    "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "الرَّحْمَٰنِ الرَّحِيمِ",
    "مَالِكِ يَوْمِ الدِّينِ",
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
    "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
]


def normalize_arabic(text: str) -> str:
    """Strip diacritics/tatweel/punctuation so ASR output can be matched
    forgivingly against the reference text."""
    if not text:
        return ""
    # remove Arabic diacritics (harakat) and tatweel
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0670\u0640]", "", text)
    # normalize alef variants, hamza seats, ya/alef maqsura, ta marbuta
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    # strip punctuation/non-arabic-letters
    text = re.sub(r"[^\u0621-\u064A\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Flat list of reference words (normalized) with ayah index, for alignment.
REFERENCE_WORDS: List[dict] = []
for ayah_idx, ayah in enumerate(FATIHA_AYAT):
    for w in ayah.split():
        REFERENCE_WORDS.append({
            "display": w,
            "norm": normalize_arabic(w),
            "ayah": ayah_idx,
        })

# --------------------------------------------------------------------------
# Model - loaded once at import time (process startup), never reloaded
# automatically. This is the key to "no restart per change" on the
# inference side: the frontend and matching logic can be iterated on
# without touching this load.
# --------------------------------------------------------------------------
log.info(f"Loading fine-tuned model {MODEL_ID} on {DEVICE} ...")
_t0 = time.time()
processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)

# Use fp16 on GPU for speed/memory, fp32 on CPU (fp16 is slow/unsupported on CPU).
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=MODEL_DTYPE
).to(DEVICE)
model.eval()

# Rebuild legacy language/task mappings from the local processor tokenizer.
vocab = processor.tokenizer.get_vocab()
model.generation_config.is_multilingual = True
model.generation_config.lang_to_id = {
    token: token_id for token, token_id in vocab.items()
    if re.fullmatch(r"<\|[a-z]{2}\|>", token)
}
model.generation_config.task_to_id = {
    "transcribe": vocab.get("<|transcribe|>"),
    "translate": vocab.get("<|translate|>"),
}
model.generation_config.no_timestamps_token_id = vocab.get("<|notimestamps|>")
log.info("Generation config patched from local tokenizer")


if DEVICE == "cuda":
    log.info(f"GPU detected: {torch.cuda.get_device_name(0)} — running in fp16")
    # Warm up CUDA kernels once at startup with a dummy silent chunk, so the
    # *first real chunk* from the user isn't slowed down by lazy CUDA init /
    # cuDNN autotuning. This keeps every subsequent chunk uniformly fast.
    try:
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        dummy_inputs = processor(dummy_audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        dummy_features = dummy_inputs.input_features.to(DEVICE).to(MODEL_DTYPE)
        with torch.no_grad():
            model.generate(dummy_features, language="arabic", task="transcribe", max_new_tokens=8)
        log.info("GPU warmup complete")
    except Exception:
        log.exception("GPU warmup failed (non-fatal, continuing)")
else:
    log.warning("No GPU detected — running on CPU. This will be noticeably slower per chunk.")

log.info(f"Model loaded and ready in {time.time() - _t0:.1f}s")


def transcribe_pcm16(pcm_bytes: bytes) -> str:
    """Transcribe raw 16-bit PCM mono @16kHz bytes -> Arabic text."""
    if len(pcm_bytes) < 2:
        return ""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(DEVICE).to(MODEL_DTYPE)
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="arabic",
            task="transcribe",
            max_new_tokens=128,
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


# Whisper models are known to "hallucinate" short generic phrases on silence
# or pure noise input instead of returning empty text. These are the most
# common hallucinated fillers for this kind of Quran-recitation checkpoint.
# If a transcript is *just* one of these (nothing else), we treat it as if
# nothing was said, rather than feeding it into word matching.
HALLUCINATION_PHRASES = {
    "الله",
    "بسم الله",
    "الله الله",
    "اللهم",
    "استغفر الله",
    "سبحان الله",
    "لا اله الا الله",
    "امين",
}


def is_probably_hallucinated(text: str) -> bool:
    """Heuristic filter: drop transcripts that are empty, extremely short,
    or match a common Whisper hallucination phrase, so they don't get fed
    into the word-matching logic as if the user actually recited them."""
    norm = normalize_arabic(text)
    if not norm:
        return True
    if norm in HALLUCINATION_PHRASES:
        return True
    # A single very short word (<=2 letters) is more likely noise/breath
    # than an actual recited word from Al-Fatiha.
    words = norm.split()
    if len(words) == 1 and len(words[0]) <= 2:
        return True
    return False


def is_speech(pcm_bytes: bytes, threshold: float = 250.0) -> bool:
    """Energy-based VAD with a stricter check than plain average RMS:
    requires BOTH the overall RMS to be above threshold AND a meaningful
    fraction of the chunk to be "active" (not just a brief click/pop).
    This avoids classifying short noise bursts or silence-with-a-blip as
    speech, which was previously causing not-yet-recited words to be
    marked wrong."""
    if len(pcm_bytes) < 2:
        return False
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(audio) == 0:
        return False

    rms = np.sqrt(np.mean(audio ** 2))
    if rms <= threshold:
        return False

    # Fraction of samples whose absolute amplitude exceeds a low activity
    # floor (half the threshold) — real speech has sustained energy across
    # a good portion of the chunk, whereas a brief pop/click doesn't.
    activity_floor = threshold * 0.5
    active_ratio = np.mean(np.abs(audio) > activity_floor)

    return bool(rms > threshold and active_ratio > 0.15)


# --------------------------------------------------------------------------
# Session state: tracks how far through Al-Fatiha each connection has
# progressed, so incremental chunks keep advancing rather than re-matching
# from the start every time.
# --------------------------------------------------------------------------
def diff_new_words(old_words: List[str], new_words: List[str]) -> List[str]:
    """Given the normalized word list from the PREVIOUS chunk's transcript and
    the normalized word list from the CURRENT chunk's transcript (which
    overlaps with the previous one, since the frontend re-sends some prior
    audio as left-context), return only the portion of `new_words` that is
    genuinely new — i.e. whatever comes after the last block that already
    matched `old_words`.

    This stops the same recited word from being fed into `apply_transcript`
    twice just because it appeared in two consecutive overlapping chunks,
    which was advancing the cursor past words that were never actually
    skipped by the reciter (e.g. marking "الرحمن" wrong/skipped after
    "الرحيم" repeated across chunk boundaries)."""
    if not old_words:
        return new_words
    sm = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    end_b = 0
    for block in sm.get_matching_blocks():
        if block.size > 0:
            end_b = max(end_b, block.b + block.size)
    return new_words[end_b:]


class RecitationSession:
    def __init__(self):
        self.cursor = 0  # index into REFERENCE_WORDS: next expected word
        self.results = ["pending"] * len(REFERENCE_WORDS)  # per-word status
        self.last_words: List[str] = []  # normalized words from the previous
        # chunk's transcript, used to diff out the overlapping portion of
        # the next chunk's transcript before matching (see diff_new_words).

    def reset(self):
        self.cursor = 0
        self.results = ["pending"] * len(REFERENCE_WORDS)
        self.last_words = []

    def apply_transcript(self, text: str) -> List[str]:
        """Diff out the overlapping portion against the previous chunk's
        transcript, then greedily align only the genuinely new words against
        the next expected reference words. Returns the genuinely-new
        (post-diff) normalized words that were actually fed into matching,
        so the caller can show the caller/frontend exactly what was used -
        not the raw overlapping chunk transcript."""
        words = [normalize_arabic(w) for w in text.split() if normalize_arabic(w)]
        new_words = diff_new_words(self.last_words, words)
        self.last_words = words
        for w in new_words:
            if self.cursor >= len(REFERENCE_WORDS):
                break
            expected = REFERENCE_WORDS[self.cursor]["norm"]
            if w == expected or _fuzzy_match(w, expected):
                self.results[self.cursor] = "correct"
                self.cursor += 1
            else:
                # Try matching against just the NEXT expected word (handles
                # ASR dropping/merging one word) - kept to a single word
                # lookahead, and an EXACT match only (no fuzzy), so a noisy
                # or garbled ASR word can't falsely "jump" the cursor two
                # words ahead and mark real, not-yet-recited words as wrong.
                idx = self.cursor + 1
                if idx < len(REFERENCE_WORDS) and w == REFERENCE_WORDS[idx]["norm"]:
                    # The skipped-over word here was NEVER actually seen in
                    # the transcript - the model may have genuinely mis-heard
                    # or dropped it (short/fast words are easy for Whisper to
                    # miss), it isn't necessarily proof the reciter said it
                    # wrong. Mark it "missed" (distinct from "wrong") instead
                    # of asserting a mispronunciation we have no evidence for.
                    self.results[self.cursor] = "missed"
                    self.results[idx] = "correct"
                    self.cursor = idx + 1
                else:
                    # Here the transcribed word actively conflicts with what
                    # was expected right now (not just absent) - this is the
                    # one case we're confident enough to call "wrong".
                    self.results[self.cursor] = "wrong"
                    self.cursor += 1
        return new_words

    def snapshot(self):
        return {
            "words": [
                {
                    "text": REFERENCE_WORDS[i]["display"],
                    "ayah": REFERENCE_WORDS[i]["ayah"],
                    "status": self.results[i],
                }
                for i in range(len(REFERENCE_WORDS))
            ],
            "cursor": self.cursor,
            "done": self.cursor >= len(REFERENCE_WORDS),
        }


def _fuzzy_match(a: str, b: str) -> bool:
    """Cheap fuzzy match: allow small edit distance for short ASR noise.
    Tightened vs. the original version: short reference words (<=3 letters,
    e.g. single-letter-ish tokens) require an exact match - a loose fuzzy
    threshold on a short word matches almost any garbled noise, which is
    what was letting noisy/short-chunk transcripts falsely "match" a future
    expected word and jump the cursor ahead of words that were never
    actually recited. The edit-distance allowance is also capped at 2
    regardless of word length, instead of growing with length."""
    if not a or not b:
        return False
    if a == b:
        return True
    if len(b) <= 3:
        return False  # too short/risky to fuzzy-match reliably
    # simple Levenshtein distance, capped for speed on short words
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return False
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    max_len = max(la, lb)
    return dp[lb] <= min(2, max(1, max_len // 3))


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(title="Tarteel Live")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    info = {"status": "ok", "model": MODEL_ID, "device": DEVICE, "dtype": str(MODEL_DTYPE)}
    if DEVICE == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
    return info


@app.get("/reference")
def reference():
    """Send the reference word list to the frontend at load time."""
    return {
        "words": [
            {"text": w["display"], "ayah": w["ayah"]} for w in REFERENCE_WORDS
        ]
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = RecitationSession()
    log.info("Client connected")

    # VAD threshold is adjustable live from the frontend without restarting.
    vad_threshold = 250.0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                pcm_bytes = message["bytes"]

                speaking = is_speech(pcm_bytes, threshold=vad_threshold)
                matched_words: List[str] = []

                if speaking:
                    try:
                        text = transcribe_pcm16(pcm_bytes)
                    except Exception as e:
                        log.exception("Transcription error")
                        text = ""
                    if text.strip() and not is_probably_hallucinated(text):
                        matched_words = session.apply_transcript(text)

                payload = {
                    "type": "update",
                    "speaking": speaking,
                    "raw_text": text if speaking else "",
                    # The exact (post-diff) words that were actually fed
                    # into matching this update - i.e. what the app treated
                    # as "newly recited", as opposed to raw_text which still
                    # includes the overlapping tail from the previous chunk.
                    "matched_words": matched_words,
                    **session.snapshot(),
                }
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))

            elif "text" in message and message["text"] is not None:
                # control messages from frontend (JSON)
                try:
                    ctrl = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue

                if ctrl.get("type") == "reset":
                    session.reset()
                    await websocket.send_text(json.dumps(
                        {"type": "update", "speaking": False, "raw_text": "",
                         "matched_words": [], **session.snapshot()}, ensure_ascii=False))

                elif ctrl.get("type") == "set_vad_threshold":
                    try:
                        vad_threshold = float(ctrl.get("value", vad_threshold))
                        log.info(f"VAD threshold set to {vad_threshold}")
                    except (TypeError, ValueError):
                        pass

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception:
        log.exception("WebSocket error")


# Serve the frontend static file(s) from the same directory for convenience.
app.mount("/static", StaticFiles(directory="."), name="static")