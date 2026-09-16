# Tarteel Live — Surah Al-Fatiha Live Recitation Checker

This app is configured to use the custom fine-tuned Quran model from the
AudioSegment project:

```text
C:\Users\CHAND COMPUTER\Desktop\AudioSegment\models\whisper-100-112-600steps
```

The backend loads this model once at startup, uses CUDA/FP16 when available,
and keeps the existing browser microphone and live word-status interface.

## Kya hai ye
  - **Backend**: `main.py` — FastAPI + WebSocket. The fine-tuned local model
  sirf **ek dafa**, process start hote waqt load hota hai (GPU pe fp16 +
  warmup ke saath). Har audio chunk pe sirf `model.generate()` call hoti hai
  — model dubara load NAHI hota. Model dubara load karne ka koi tareeqa nahi
  except process restart — is liye `uvicorn` ko bina `--reload` ke chalayein
  (neeche dekhein), taake baar baar model reload na ho.
- **Frontend**: `index.html` — mic se live audio capture, PCM16 @16kHz mein
  convert karke websocket se chunks bhejta hai. User chunk size aur VAD
  sensitivity live adjust kar sakta hai (koi restart nahi chahiye).

## Setup (pehli dafa)

```bash
cd tarteel-live
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU**: Code apne aap GPU detect kar leta hai (`torch.cuda.is_available()`)
> aur agar mile to fp16 mein model load karta hai (fast + kam VRAM). Kuch
> nahi karna padta manually — bas ye zaroori hai ke aapka `torch` install
> CUDA-wala ho, default pip wala nahi. Pehle CUDA-enabled torch install
> karein apne CUDA version ke mutabiq (dekhein
> https://pytorch.org/get-started/locally/, e.g.:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`),
> phir baaki requirements. Startup pe backend GPU ka naam log karega aur
> ek chhota "warmup" call chalayega taake pehla real chunk bhi fast ho.
> Confirm karne ke liye `http://localhost:8000/health` kholein — ismein
> `device`, `gpu_name` aur `dtype` dikhega.

## Chalane ka tareeqa

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

From this project directory, the current command is:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

To use another local model without editing the code:

```powershell
$env:TARTEEL_MODEL_PATH = "C:\path\to\model"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

**`--reload` MAT lagayein.** Reload flag har file-save pe process restart
karta hai jo model ko dubara load karwa dega (30–60+ sec ka overhead har
dafa). Is setup mein aapko reload ki zaroorat hi nahi:
- Frontend (`index.html`) mein koi bhi change seedha browser refresh (F5) se
  reflect ho jata hai — backend restart ki zaroorat nahi.
- Agar sirf matching-logic (`RecitationSession`, VAD threshold, wagera) mein
  chhota change karna ho, wo bhi backend.py mein hi hai — us surat mein sirf
  ek dafa `Ctrl+C` karke phir se upar wala command chalayein. Model load
  poora dobara hoga lekin sirf tab jab aap khud restart karein, automatic
  nahi.

Server chalne ke baad browser mein kholein:

```
http://localhost:8000/
```

(Agar kisi doosre device/phone se test karna ho to `http://<is-machine-ka-IP>:8000/`
use karein — mic permission ke liye HTTPS ya `localhost` chahiye hota hai,
LAN pe plain HTTP se remote device pe mic block ho sakta hai. Chrome mein
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` se us URL ko
whitelist kar sakte hain testing ke liye.)

## Frontend controls
- **Chunk size slider (500ms–3000ms, default 1750ms)**: kitni der ka audio
  ikattha karke ek dafa mein backend ko bheja jaye. Chota = zyada responsive
  lekin kam context/accuracy. Bada = zyada accurate lekin thoda lag mehsoos
  hoga.
- **VAD threshold slider**: silence vs speech detect karne ki sensitivity.
  Agar background noise zyada hai to badhayein; agar aahista bolne pe bhi
  "silence" dikh raha hai to kam karein. Ye live backend ko bhejta hai, koi
  restart nahi chahiye.
- **Start / Stop**: mic capture on/off.
- **Reset**: Fatiha progress dobara shuru se.

## Kaise kaam karta hai (short version)
1. Browser mic se audio capture karta hai, 16kHz PCM16 mein resample karta
   hai (webm/opus ki jagah — decode overhead nahi, seedha model-ready format).
2. Har chunk websocket se backend ko jata hai.
3. Backend energy-based VAD se check karta hai speech hai ya silence
   (green/red indicator).
4. Agar speech hai to Whisper-tiny-ar-quran se transcribe karta hai.
5. Transcript ko Surah Al-Fatiha ke expected words ke against greedy +
   fuzzy (chhoti edit-distance) matching se align karta hai.
6. Result har word ke liye status bhejta hai: `correct` (green),
   `wrong` (red), `pending` (grey), aur current expected word highlight
   (yellow) hota hai.

## Known limitations / next steps (agar zaroorat pade)
- VAD abhi simple RMS-energy based hai (fast, koi extra dependency nahi).
  Agar zyada robust chahiye to `webrtcvad` add kar sakte hain.
- Matching greedy hai — agar reciter koi lafz repeat kare ya beech mein ruk
  kar wapas jaye, alignment thoda confuse ho sakta hai. Is scope ke liye
  (testing) theek hai; production ke liye Needleman-Wunsch jaisa proper
  sequence alignment better hoga.
- `ScriptProcessorNode` deprecated hai but sab browsers mein chalta hai;
  agar chahein to `AudioWorklet` mein upgrade kar sakte hain baad mein.
