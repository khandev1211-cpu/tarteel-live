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
import json
import re
import wave
import time
import logging
from typing import List

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
MODEL_ID = "tarteel-ai/whisper-tiny-ar-quran"
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
log.info(f"Loading {MODEL_ID} on {DEVICE} ...")
_t0 = time.time()
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Use fp16 on GPU for speed/memory, fp32 on CPU (fp16 is slow/unsupported on CPU).
MODEL_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=MODEL_DTYPE
).to(DEVICE)
model.eval()

# This checkpoint ships an outdated generation_config.json that predates the
# `language=`/`task=` kwargs in generate(). Swapping in a current base-Whisper
# generation config (same architecture/tokenizer family) fixes this
# permanently, one time at load — not something done per chunk.
try:
    base_gen_config = GenerationConfig.from_pretrained("openai/whisper-tiny")
    model.generation_config = base_gen_config
    log.info("Patched generation_config from openai/whisper-tiny (fixes outdated config)")
except Exception:
    log.exception("Could not patch generation_config; language=/task= may fail")


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
            model.generate(dummy_features, language="ar", task="transcribe", max_new_tokens=8)
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
            language="ar",
            task="transcribe",
            max_new_tokens=128,
        )
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text


def is_speech(pcm_bytes: bytes, threshold: float = 250.0) -> bool:
    """Simple energy-based VAD: RMS amplitude above threshold = speech.
    threshold is on int16 scale; tune from the frontend if needed."""
    if len(pcm_bytes) < 2:
        return False
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2)) if len(audio) else 0.0
    return rms > threshold


# --------------------------------------------------------------------------
# Session state: tracks how far through Al-Fatiha each connection has
# progressed, so incremental chunks keep advancing rather than re-matching
# from the start every time.
# --------------------------------------------------------------------------
class RecitationSession:
    def __init__(self):
        self.cursor = 0  # index into REFERENCE_WORDS: next expected word
        self.results = ["pending"] * len(REFERENCE_WORDS)  # per-word status

    def reset(self):
        self.cursor = 0
        self.results = ["pending"] * len(REFERENCE_WORDS)

    def apply_transcript(self, text: str):
        """Greedy word-by-word alignment of newly transcribed words against
        the next expected reference words."""
        words = [normalize_arabic(w) for w in text.split() if normalize_arabic(w)]
        for w in words:
            if self.cursor >= len(REFERENCE_WORDS):
                break
            expected = REFERENCE_WORDS[self.cursor]["norm"]
            if w == expected or _fuzzy_match(w, expected):
                self.results[self.cursor] = "correct"
                self.cursor += 1
            else:
                # try matching against the next couple of expected words
                # (handles ASR dropping/merging a word)
                matched_ahead = False
                for lookahead in (1, 2):
                    idx = self.cursor + lookahead
                    if idx < len(REFERENCE_WORDS) and (
                        w == REFERENCE_WORDS[idx]["norm"]
                        or _fuzzy_match(w, REFERENCE_WORDS[idx]["norm"])
                    ):
                        for skip in range(self.cursor, idx):
                            self.results[skip] = "wrong"
                        self.results[idx] = "correct"
                        self.cursor = idx + 1
                        matched_ahead = True
                        break
                if not matched_ahead:
                    self.results[self.cursor] = "wrong"
                    self.cursor += 1

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
    """Cheap fuzzy match: allow small edit distance for short ASR noise."""
    if not a or not b:
        return False
    if a == b:
        return True
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
    return dp[lb] <= max(1, max_len // 3)


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

                if speaking:
                    try:
                        text = transcribe_pcm16(pcm_bytes)
                    except Exception as e:
                        log.exception("Transcription error")
                        text = ""
                    if text.strip():
                        session.apply_transcript(text)

                payload = {
                    "type": "update",
                    "speaking": speaking,
                    "raw_text": text if speaking else "",
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
                         **session.snapshot()}, ensure_ascii=False))

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