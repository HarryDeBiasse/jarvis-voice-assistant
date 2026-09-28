#!/usr/bin/env python3
"""Jerry - in-room voice assistant.

Pipeline: wake('hey_jerry') -> record-until-endpoint(Silero VAD) -> STT(Deepgram stream, Whisper fallback)
          -> route(instant-command | fast Haiku | OpenClaw brain) -> TTS(Piper, sentence-streamed) -> speaker.

Hardware (verified 2026-09-10):
  MIC  = ReSpeaker XVF3800 (ALSA card 3), native S16_LE / 16000 Hz / 2ch.
  SPKR = 3.5mm jack (ALSA card 0, bcm2835 Headphones) -> aux -> Alexa.

CLI:
  jerry_voice.py                 run the full wake loop
  jerry_voice.py --mic           record 3s, transcribe, print (mic+STT test)
  jerry_voice.py --say "text"    speak text (TTS test)
  jerry_voice.py --ask "text"    route+answer one line (no audio in)
"""
import os, sys, re, json, time, wave, tempfile, subprocess, random, threading
import numpy as np
import yaml

HOME = os.path.expanduser("~")
CFG = yaml.safe_load(open(os.path.join(HOME, "jarvis", "config.yaml")))

# --- audio device constants (from verified hardware) ---
MIC_CARD = 3            # ReSpeaker XVF3800
MIC_RATE = 16000
MIC_CH   = 2            # device is stereo; we use channel 0 (beamformed)
SPK_DEV  = "plughw:CARD=Array,DEV=0"  # XVF3800 array out -> Alexa AND AEC far-end reference (hardware echo cancel)
FRAME    = 1280         # 80ms @ 16k, openWakeWord's expected chunk

WAKE_THRESHOLD = 0.5
# adaptive endpoint: Silero VAD gives RELIABLE silence detection, so we end fast on short
# turns and give longer/complex turns (which tend to pause mid-thought) more grace.
HANG_SHORT     = 0.55   # utterance < 2s (a quick command/confirmation) — trimmed for snappier feel
HANG_NORM      = 0.85   # normal utterance
HANG_LONG      = 1.6    # utterance > 12s (deep in a thought — don't cut off)
# barge-in: stop talking the moment the user speaks over us (relies on the XVF3800 hardware AEC
# cancelling our own voice; guarded by a HIGH VAD confidence + sustained duration so residual echo
# can't self-interrupt). Flip BARGE_IN=False to instantly fall back to old uninterruptible speech.
BARGE_IN       = False  # TEMP off: relaxed mute-during-speech caused a self-hearing feedback loop; re-enable after fix
BARGE_VAD      = 0.85   # VAD speech-prob required DURING our own speech (well above the 0.5 capture gate)
BARGE_CHUNKS   = 4      # consecutive voiced chunks (~320ms, spec §5.2 ~250ms) before it counts as a real interruption
# transcript-keyed endpointing (§3.2/3.3): flex the silence-hang on the LAST word of the live partial.
# NOTE: trimmed from the spec's fuller list — dropped common sentence-ENDERS (it/is/the/a/on/of/in/at/for…)
# because they wrongly slowed complete requests ("what time is IT", "turn the lights ON"). Kept only words
# that almost never end a real request, so grace is given for genuine trailing ("…and", "…because", "…to").
_CONT_WORDS = set(("and or but so because if when while with that which to like just really "
                   "kind sort maybe probably about my your their").split())
_FILLER_WORDS = set("um uh er erm hmm uhh umm".split())
_CONFIRM_WORDS = set("yeah yes no ok okay right sure stop nope yep yup nah".split())
def _adaptive_hang(partial, dur):
    w = re.sub(r"[^a-z' ]", "", (partial or "").lower()).split()
    last = w[-1] if w else ""
    if last in _FILLER_WORDS:  return 1.8            # trailing "um…" — they're thinking, wait
    if last in _CONT_WORDS:    return 1.4            # "…and", "…the" — mid-sentence, give grace
    if len(w) <= 2 and last in _CONFIRM_WORDS: return 0.35   # "yeah"/"ok" — end fast
    return HANG_SHORT if dur < 2.0 else (HANG_LONG if dur > 12.0 else HANG_NORM)
def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p.strip()] or [text]
MAX_UTTER      = 20.0   # hard cap on a single utterance (s) - generous so pauses fit
FOLLOWUP_WINDOW = 8.0   # seconds mic stays open after EVERY reply for a follow-up (no wake word)
FOLLOWUP_ONSET  = 550   # RMS to START a follow-up (well above the noise floor, ignores echo/noise)
REQUEST_ONSET   = 450   # RMS for the user to START their request after the wake ack
REQUEST_WAIT    = 8.0   # seconds to wait (silently) for a request after the ack before giving up
POST_SPEAK_FLUSH = 0.5   # brief tail-swallow; hardware AEC handles the echo so the follow-up window stays responsive
WAKE_ACK_DELAY  = (0.0, 0.0) # instant ack (was 0.4-0.9s "human beat" — pure lag; set e.g. (0.1,0.3) to add a beat back)
# spoken right after a request lands, while the brain works (varied = less robotic)
FILLERS = [
    "On it.", "Right away.", "One moment.", "Right away, sir.", "Let me see to that.",
    "Working on it.", "Looking into it now.", "Give me a moment.", "Certainly.",
    "On it, sir.", "Let me check that.", "Just a moment.", "Of course.",
    "Allow me a moment.", "Checking now.", "Let me find out.",
]
# spoken the instant the wake word fires (pre-synthesized so it's instant)
WAKE_ACKS = [
    "Yes, sir?", "Sir?", "At your service.", "Yes?", "How can I help?",
    "Standing by.", "Go ahead.", "Listening.",
]

# ---------------------------------------------------------------- TTS (WARM Piper)
# Piper's CLI reloads its 63MB model on every call (~6-7s cold). We load the model
# ONCE via the Python API and reuse it, so dynamic replies synth in ~0.5-1s.
from piper import PiperVoice
_voice = None
def _get_voice():
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(CFG["piper_voice"])   # cold load ~7s, ONCE at startup
    return _voice

def _synth(text, wav_path):
    with wave.open(wav_path, "wb") as wf:
        _get_voice().synthesize_wav(text, wf)

# HALF-DUPLEX: the mic is ignored until this timestamp, so Jarvis never hears
# its own voice coming back off the Alexa (no AEC reference across devices).
_mute_until = 0.0
def _wav_dur(path):
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate() or 16000)
    except Exception:
        return 1.5

def _play(wav_path, cooldown=1.0):
    global _mute_until
    # go deaf for the whole playback + a cooldown so the room echo dies out first
    _mute_until = time.time() + _wav_dur(wav_path) + cooldown
    subprocess.run(["aplay", "-q", "-D", SPK_DEV, wav_path], check=False)
    _mute_until = time.time() + cooldown   # re-anchor from actual playback end

def speak(text, cooldown=1.2):
    if not text:
        return
    wav = tempfile.mktemp(suffix=".wav")
    try:
        _synth(text, wav)        # warm synth ~0.5-1s
        _play(wav, cooldown)
    finally:
        if os.path.exists(wav):
            os.remove(wav)

def _presynth(phrases, subdir):
    """Pre-render a fixed phrase set to cached wavs so playback is instant."""
    d = os.path.join(HOME, "jarvis", subdir)
    os.makedirs(d, exist_ok=True)
    wavs = []
    for i, ph in enumerate(phrases):
        p = os.path.join(d, f"{subdir}{i}.wav")
        if not os.path.exists(p):
            _synth(ph, p)
        wavs.append(p)
    return wavs

_ack_wavs, _filler_wavs = [], []
def ensure_ack():
    global _ack_wavs, _filler_wavs
    _ack_wavs = _presynth(WAKE_ACKS, "acks")
    _filler_wavs = _presynth(FILLERS, "fillers")

def play_ack():
    if _ack_wavs:
        _play(random.choice(_ack_wavs), cooldown=0.25)   # short: user speaks right after

def play_filler_async():
    """Non-blocking filler: starts playback and returns the process immediately, so the
    ~2.4s STT runs DURING the filler instead of after it. Returns the Popen (or None)."""
    global _mute_until
    if not _filler_wavs:
        return None
    wav = random.choice(_filler_wavs)
    _mute_until = time.time() + _wav_dur(wav) + 0.4   # stay deaf through the filler
    try:
        return subprocess.Popen(["aplay", "-q", "-D", SPK_DEV, wav])
    except Exception:
        return None

# ---------------------------------------------------------------- instant commands
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None
def _now():
    return datetime.now(_TZ) if _TZ else datetime.now()

def wake_pc():
    subprocess.run(["wakeonlan", CFG["pc_mac"]], check=False)

# Locally-answered intents that SKIP the ~6s brain round-trip (answered in <1s).
INSTANT = [
    (re.compile(r"turn on (my )?(pc|computer|desktop)", re.I),
     lambda: (wake_pc(), "On it, sir. Booting your PC now.")),
    (re.compile(r"\btime is it\b|\bwhat time\b|\bthe time\b|\bcurrent time\b|what'?s the time|\btime right now\b", re.I),
     lambda: _now().strftime("It's %-I:%M %p.")),
    (re.compile(r"what'?s the date|what is the date|what day is it|today'?s date|\bthe date\b|\bdate is it\b|\bwhat day\b|\bwhat'?s today\b", re.I),
     lambda: _now().strftime("It's %A, %B %-d.")),
]

# ---------------------------------------------------------------- brain link (tiered)
# Tier 1 (fast): direct Anthropic API, Haiku — quick answers ~1-2s, pennies.
# Tier 2 (deep): the full Jerry agent via the gateway tunnel — tools + memory, ~6s.
BRAIN_SYS = ("You are Jerry, Harry's in-room voice assistant — composed, capable, and unflappable in the "
             "style of JARVIS from Iron Man, with dry wit sprinkled in occasionally, not every line. "
             "Your name is Jerry; if asked who you are, you are Jerry (never Jarvis). "
             "Address him as 'sir' now and then, never in every sentence. Reply by VOICE in ONE or two "
             "short spoken sentences; lead with the answer or action and stay crisp. "
             "No markdown, lists, or code - this is read aloud.")
FAST_SYS = (BRAIN_SYS + " If the request needs live or current info (weather, news, prices, "
            "the date/time), web access, files, the PC, or any multi-step work you cannot answer "
            "confidently from general knowledge in one breath, reply with EXACTLY one word: ESCALATE")
THINKING = ["Let me look into that, sir.", "One moment while I work that out.",
            "That'll take a moment - let me dig in.", "Allow me a moment on that.",
            "Looking into it now, sir."]

FAST_MODEL = "claude-haiku-4-5"
# USD per 1M tokens (input, output) — from the claude-api reference (2026)
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (3.0, 15.0),
         "claude-opus-4-8": (5.0, 25.0)}
_KEY_FILE = os.path.join(HOME, "jarvis", ".anthropic_key")
_COST_LOG = os.path.join(HOME, "jarvis", "api_cost.json")

def _api_key():
    try:
        k = open(_KEY_FILE).read().strip()
        return k or None
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")

def _log_cost(model, usage):
    pin, pout = PRICE.get(model, (0.0, 0.0))
    itok = usage.get("input_tokens", 0); otok = usage.get("output_tokens", 0)
    cost = itok / 1e6 * pin + otok / 1e6 * pout
    try:
        d = json.load(open(_COST_LOG))
    except Exception:
        d = {"total_cost": 0.0, "total_in": 0, "total_out": 0, "calls": 0, "by_day": {}}
    day = _now().strftime("%Y-%m-%d")
    d["total_cost"] += cost; d["total_in"] += itok; d["total_out"] += otok; d["calls"] += 1
    dd = d["by_day"].setdefault(day, {"cost": 0.0, "calls": 0})
    dd["cost"] += cost; dd["calls"] += 1
    try:
        json.dump(d, open(_COST_LOG, "w"))
    except Exception:
        pass

def ask_fast(text):
    """Direct Anthropic API (Haiku). Returns (answer | None, escalate_bool)."""
    import urllib.request, urllib.error
    key = _api_key()
    if not key:
        return None, True                        # no key -> use the deep brain
    body = json.dumps({
        "model": FAST_MODEL, "max_tokens": 200, "system": FAST_SYS,
        "messages": [{"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
        method="POST", headers={"content-type": "application/json",
        "x-api-key": key, "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        txt = "".join(b.get("text", "") for b in d.get("content", [])).strip()
        _log_cost(FAST_MODEL, d.get("usage", {}))
        if re.sub(r"[^a-z]", "", txt.lower()) == "escalate":
            return None, True
        return txt, False
    except Exception as e:
        print("[jerry] fast api error:", e)
        return None, True                        # fall back to the deep brain

def _gateway_token():
    try:
        c = json.load(open(os.path.join(HOME, ".openclaw", "openclaw.json")))
        return c.get("gateway", {}).get("auth", {}).get("token")
    except Exception:
        return None

def ask_brain(text):
    """Full Jerry agent via the gateway tunnel (tools + memory, ~6s)."""
    import urllib.request, urllib.error
    tok = _gateway_token()
    url = CFG["openclaw_gateway"].rstrip("/") + "/v1/chat/completions"
    body = json.dumps({
        "model": "openclaw/default", "user": "jerry-voice-room",
        "messages": [{"role": "system", "content": BRAIN_SYS},
                     {"role": "user", "content": text}],
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}",
                 "x-openclaw-model": "anthropic/claude-sonnet-5"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        return d["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "My brain link isn't switched on yet."
        return f"Brain link error {e.code}."
    except Exception:
        return "I couldn't reach my brain just now."

# §4.4 local escalation heuristic — cheap, no model call: does this look like a 'deep' question?
_DEEP_RX = re.compile(r"\b(why|how (do|can|would|should) (i|we|you)|explain|compare|design|plan|"
                      r"write|code|debug|should i|difference between|what'?s the difference|"
                      r"walk me through|help me|summari[sz]e|pros and cons|what do you think)\b", re.I)
def _looks_deep(text):
    return bool(_DEEP_RX.search(text)) or len(text.split()) > 25

def handle(text):
    # 1) local instant intents (time/date/PC) — <1s, no model
    for rx, act in INSTANT:
        if rx.search(text):
            r = act()
            return r[1] if isinstance(r, tuple) else r
    # 2) speculative dual dispatch (§4): always start the fast tier; if it LOOKS deep, fire the
    #    brain IN PARALLEL so we skip the fast-decides-then-escalate serial penalty.
    res = {}
    def _f(): res["fast"] = ask_fast(text)
    def _d(): res["deep"] = ask_brain(text)
    tf = threading.Thread(target=_f, daemon=True); tf.start()
    if _looks_deep(text):
        td = threading.Thread(target=_d, daemon=True); td.start()
        speak(random.choice(THINKING))          # spoken bridge — don't wait on fast, cover the brain
        td.join()
        if res.get("deep"):
            return res["deep"]
        tf.join()                               # brain failed → fall back to the fast answer
        return (res.get("fast") or (None,))[0] or "I couldn't work that out just now."
    # 3) not obviously deep: let the fast tier answer; escalate only if it bails
    tf.join()
    answer, escalate = res.get("fast", (None, True))
    if answer and not escalate:
        return answer
    td = threading.Thread(target=_d, daemon=True); td.start()
    speak(random.choice(THINKING))
    td.join()
    return res.get("deep") or answer or "I couldn't work that out just now."

# ---------------------------------------------------------------- STT
_whisper = None
def stt(int16_mono):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel(CFG["whisper_model"], device="cpu",
                                compute_type="int8", cpu_threads=4)
    audio = (int16_mono.astype(np.float32)) / 32768.0
    segs, _ = _whisper.transcribe(audio, language="en", beam_size=1,
                                  vad_filter=True)
    return "".join(s.text for s in segs).strip()

# ---------------------------------------------------------------- streaming STT (Deepgram)
def _dg_key():
    try:
        k = open(os.path.join(HOME, "jarvis", ".deepgram_key")).read().strip()
        return k or None
    except Exception:
        return os.environ.get("DEEPGRAM_API_KEY")

class DeepgramStream:
    """Live streaming STT over Deepgram's websocket. Audio is streamed WHILE the user talks,
    so the transcript is ready ~instantly at endpoint (~150-300ms) instead of a ~2.4s local
    Whisper pass. We control endpointing ourselves (Silero), so Deepgram's is disabled.
    Any failure (no key, connect/socket error, empty result) → caller falls back to local Whisper."""
    URL = ("wss://api.deepgram.com/v1/listen?model=nova-2&encoding=linear16"
           "&sample_rate=16000&channels=1&interim_results=true&endpointing=false&punctuate=true")
    def __init__(self, key):
        self.key = key; self.ws = None; self.active = False
        self._final = []; self._interim = ""; self._lock = threading.Lock(); self._rx = None
    def start(self):
        self._final = []; self._interim = ""; self.active = False; self.ws = None
        if not self.key:
            return False
        try:
            import websocket
            ws = websocket.create_connection(self.URL, timeout=5,
                    header=["Authorization: Token %s" % self.key])
            ws.settimeout(5)
            self.ws = ws; self.active = True
            self._rx = threading.Thread(target=self._recv, daemon=True); self._rx.start()
            return True
        except Exception as e:
            print("[jerry] deepgram connect failed:", e); self.active = False; return False
    def _recv(self):
        while self.active:
            try:
                msg = self.ws.recv()
            except Exception:
                break
            if not msg:
                continue
            try:
                d = json.loads(msg)
            except Exception:
                continue
            alt = (d.get("channel", {}).get("alternatives") or [{}])[0]
            t = (alt.get("transcript") or "").strip()
            if not t:
                continue
            with self._lock:
                if d.get("is_final"):
                    self._final.append(t)
                else:
                    self._interim = t
    def send(self, pcm_int16):
        if self.active and self.ws is not None:
            try:
                self.ws.send_binary(pcm_int16.tobytes())
            except Exception:
                self.active = False
    def current_text(self):
        """Live partial transcript so far (finals + latest interim) — used for §3.2 endpoint timing."""
        with self._lock:
            return (" ".join(self._final) + " " + self._interim).strip()
    def finish(self, wait=1.2):
        if not self.ws:
            return ""
        try:
            self.ws.send(json.dumps({"type": "CloseStream"}))   # flush remaining audio -> finals
        except Exception:
            pass
        end = time.time() + wait; last = None
        while time.time() < end:      # wait (briefly) for finals to stop growing
            with self._lock:
                cur = " ".join(self._final).strip()
            if cur and cur == last:
                break
            last = cur; time.sleep(0.05)
        self.active = False
        try:
            self.ws.close()
        except Exception:
            pass
        with self._lock:
            return (" ".join(self._final).strip() or self._interim.strip())

# ---------------------------------------------------------------- VAD (Silero v5, ONNX)
VAD_MODEL     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "silero_vad.onnx")
VAD_THRESHOLD = 0.5     # Silero speech probability above this = voiced
VAD_WINDOW    = 512     # Silero v5 requires exactly 512 samples @ 16kHz per call (32ms)

class SileroVAD:
    """Silero VAD v5 over onnxruntime. Feed arbitrary int16 chunks; it buffers them into
    512-sample windows and reports whether a chunk contained voiced speech. Far more robust
    than an RMS energy gate — ignores fans, echo tails and room noise, so we can endpoint
    on short, reliable silences instead of a blunt fixed hang."""
    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.inter_op_num_threads = 1; so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(VAD_MODEL, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self._sr = np.array(16000, dtype=np.int64)
        self.reset()
    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
    def voiced(self, chunk_int16, thresh=None):
        th = VAD_THRESHOLD if thresh is None else thresh
        x = chunk_int16.astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x
        voiced = False
        while self._buf.size >= VAD_WINDOW:
            frame = self._buf[:VAD_WINDOW].reshape(1, VAD_WINDOW)
            self._buf = self._buf[VAD_WINDOW:]
            out, self._state = self.sess.run(None, {"input": frame, "state": self._state, "sr": self._sr})
            self.last_prob = float(out[0][0])
            if self.last_prob >= th:
                voiced = True
        return voiced

# ---------------------------------------------------------------- audio capture
def _open_stream():
    import sounddevice as sd
    # find the ReSpeaker input device index by name
    idx = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] >= MIC_CH and "reSpeaker" in d["name"]:
            idx = i; break
    return sd, idx

# ---------------------------------------------------------------- wake loop
def run():
    import sounddevice as sd, queue
    from openwakeword.model import Model
    import openwakeword
    wpath = os.path.join(os.path.expanduser("~"), "jarvis", "hey_jerry.onnx")
    oww = Model(wakeword_model_paths=[wpath])
    ensure_ack()
    idx = None
    while idx is None:
        _, idx = _open_stream()
        if idx is None:
            print("[jerry] ReSpeaker mic not found - waiting 5s (replug the USB mic)...", flush=True)
            time.sleep(5)
            try: sd._terminate(); sd._initialize()   # force PortAudio to re-scan USB
            except Exception: pass
    print(f"[jerry] listening for wake word on device {idx} ... (say 'hey jerry')")

    q = queue.Queue()
    def cb(indata, frames, t, status):
        if time.time() < _mute_until:   # HALF-DUPLEX: drop mic while/just-after Jarvis speaks
            return
        q.put(indata[:, 0].copy())

    def drain():
        while not q.empty():
            try: q.get_nowait()
            except Exception: break

    def rms_of(c):
        return float(np.sqrt(np.mean(c.astype(np.float32) ** 2)))
    def has_words(t):
        return len(re.sub(r"[^a-z0-9]", "", t.lower())) >= 2   # ignore ""/"."/noise
    def flush(sec):
        # discard mic input for `sec` seconds (kills self-echo after we speak)
        deadline = time.time() + sec
        while time.time() < deadline:
            try: q.get(timeout=0.1)
            except Exception: pass

    vad = SileroVAD()   # neural voice-activity detector for adaptive endpointing
    bvad = SileroVAD()  # separate VAD for barge-in detection while WE are speaking
    dg = DeepgramStream(_dg_key())   # streaming STT; falls back to local whisper on any failure

    def _play_wav_interruptible(wpath):
        """Play one wav, watching for a barge-in. Returns True if the user cut in."""
        try:
            proc = subprocess.Popen(["aplay", "-q", "-D", SPK_DEV, wpath])
        except Exception:
            return False
        voiced = 0; interrupted = False
        while proc.poll() is None:
            try:
                c = q.get(timeout=0.2)
            except queue.Empty:
                continue
            if bvad.voiced(c, BARGE_VAD):
                voiced += 1
                if voiced >= BARGE_CHUNKS:
                    interrupted = True
                    try: proc.terminate()
                    except Exception: pass
                    break
            else:
                voiced = 0
        try: proc.wait(timeout=1)
        except Exception:
            try: proc.kill()
            except Exception: pass
        return interrupted

    def speak_interruptible(text):
        """Speak the reply SENTENCE-BY-SENTENCE (§5.6): first audio starts as soon as sentence 1 is
        synthesized, while later sentences synthesize in the background — so a long reply doesn't wait
        for the whole synth. Keeps barge-in per sentence: sustained high-confidence speech over us stops
        playback at once. Returns True if interrupted. BARGE_IN off → whole-reply blocking speech."""
        global _mute_until
        if not text:
            return False
        if not BARGE_IN:
            wav = tempfile.mktemp(suffix=".wav")
            try: _synth(text, wav); _play(wav, 1.2)
            except Exception: pass
            try:
                if os.path.exists(wav): os.remove(wav)
            except Exception: pass
            return False
        sents = _split_sentences(text)
        wavq = queue.Queue(); stop = threading.Event()
        def producer():
            for s in sents:
                if stop.is_set():
                    break
                w = tempfile.mktemp(suffix=".wav")
                try:
                    _synth(s, w); wavq.put(w)
                except Exception:
                    pass
            wavq.put(None)   # sentinel: done
        threading.Thread(target=producer, daemon=True).start()
        _mute_until = 0.0            # do NOT hard-mute — we must be able to hear a barge-in
        drain(); bvad.reset()
        interrupted = False
        while True:
            w = wavq.get()          # wait for the next synthesized sentence (or sentinel)
            if w is None:
                break
            interrupted = _play_wav_interruptible(w)
            try: os.remove(w)
            except Exception: pass
            if interrupted:
                stop.set()
                break
        while True:                 # discard any sentences synthesized after a barge-in
            try: w = wavq.get_nowait()
            except queue.Empty:
                break
            if w and os.path.exists(w):
                try: os.remove(w)
                except Exception: pass
        _mute_until = time.time() + (0.15 if interrupted else 0.6)
        return interrupted

    # ONE stream for wake detect + capture + follow-up (device is single-open)
    state = "listen"
    buf, started, last_voice, follow_deadline, onset, heard_speech = [], 0.0, 0.0, 0.0, 0, False
    with sd.InputStream(device=idx, samplerate=MIC_RATE, channels=MIC_CH,
                        dtype="int16", blocksize=FRAME, callback=cb):
        while True:
            try:
                chunk = q.get(timeout=8)
            except queue.Empty:
                if time.time() < _mute_until:
                    continue                      # muted on purpose (Jarvis speaking)
                raise RuntimeError("no audio for 8s - mic likely disconnected")
            now = time.time()

            if state == "listen":
                scores = oww.predict(chunk)
                score = scores.get("hey_jerry", 0)
                if score > WAKE_THRESHOLD:
                    print(f"[jerry] WAKE ({score:.2f})")
                    lo, hi = WAKE_ACK_DELAY
                    if hi > 0: time.sleep(lo + random.random()*(hi-lo))
                    play_ack()
                    oww.reset(); drain()          # drop ack echo + backlog
                    buf = []; heard_speech = False; started = last_voice = time.time()
                    state = "record"

            elif state == "followup":
                # mic stays open after a reply: grab a follow-up with NO wake word.
                # Require SUSTAINED loud speech (2+ chunks over a high threshold) so
                # residual echo / room noise can't self-trigger a phantom question.
                if rms_of(chunk) > FOLLOWUP_ONSET:
                    onset += 1
                else:
                    onset = 0
                if onset >= 2:
                    print("[jerry] follow-up detected")
                    vad.reset(); dg.start(); dg.send(chunk); buf = [chunk]; heard_speech = True; started = last_voice = now; onset = 0
                    state = "record"
                elif now > follow_deadline:
                    print("[jerry] follow-up window closed -> wake word")
                    oww.reset(); drain()
                    state = "listen"

            else:  # record: wait for the request, THEN capture until trailing silence
                if not heard_speech:
                    # after the wake ack, wait SILENTLY for the user to actually start talking
                    if rms_of(chunk) > REQUEST_ONSET:
                        vad.reset(); dg.start(); dg.send(chunk); heard_speech = True; buf = [chunk]; started = last_voice = now
                    elif now - started > REQUEST_WAIT:
                        print("[jerry] woke but no request -> back to wake word")
                        oww.reset(); drain(); state = "listen"
                    continue
                buf.append(chunk)
                dg.send(chunk)
                if vad.voiced(chunk):
                    last_voice = now
                dur = now - started
                hang = _adaptive_hang(dg.current_text() if dg.active else "", dur)
                done = (now - last_voice > hang and dur > 0.3) \
                       or (dur > MAX_UTTER)
                if done:
                    utt = np.concatenate(buf) if buf else np.zeros(0, np.int16)
                    fp = play_filler_async()       # non-blocking filler covers any residual gap
                    text = dg.finish() if dg.active else ""   # streaming transcript — ready ~instantly
                    if not (text and has_words(text)):
                        text = stt(utt)            # fallback: local Whisper if Deepgram missed/failed
                    if fp:
                        try: fp.wait(timeout=4)
                        except Exception: pass
                    print(f"[jerry] heard: {text!r}")
                    if text and has_words(text):
                        reply = handle(text)
                        print(f"[jerry] reply: {reply!r}")
                        interrupted = speak_interruptible(reply)
                        oww.reset(); drain()
                        if interrupted:
                            print("[jerry] barge-in — stopped for you")
                        else:
                            flush(POST_SPEAK_FLUSH)    # swallow the reply echo tail
                        follow_deadline = time.time() + FOLLOWUP_WINDOW
                        onset = 0
                        state = "followup"   # 1st turn needs "Hey Jerry"; then 8s follow-up window (no wake word). Adaptive hang applies here too.
                    else:
                        print("[jerry] no real words -> back to wake word")
                        oww.reset(); drain()
                        state = "listen"

def run_forever():
    """Run the wake loop; on any audio/stream error (e.g. the USB mic dropping),
    re-scan the devices and restart instead of crashing — so an unplug/replug or
    a flaky connection self-heals within a few seconds."""
    import sounddevice as sd
    while True:
        try:
            run()
        except Exception as e:
            print(f"[jerry] stream error ({type(e).__name__}: {e}) - recovering in 3s", flush=True)
            try: sd._terminate(); sd._initialize()
            except Exception: pass
            time.sleep(3)

# ---------------------------------------------------------------- CLI
def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--say":
        speak(sys.argv[2] if len(sys.argv) > 2 else "Hello, I am Jerry."); return
    if len(sys.argv) >= 2 and sys.argv[1] == "--ask":
        print(handle(sys.argv[2])); return
    if len(sys.argv) >= 2 and sys.argv[1] == "--mic":
        sd, idx = _open_stream()
        if idx is None:
            print("ReSpeaker not found"); return
        print("recording 3s ...")
        import sounddevice as sd2
        rec = sd2.rec(int(3 * MIC_RATE), samplerate=MIC_RATE, channels=MIC_CH,
                      dtype="int16", device=idx); sd2.wait()
        print("transcribing ...")
        print("HEARD:", repr(stt(rec[:, 0])))
        return
    run_forever()

if __name__ == "__main__":
    main()
