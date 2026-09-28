# Jerry — Offline-First In-Room Voice Assistant

A low-latency, JARVIS-style voice assistant that runs on a Raspberry Pi: say
**"Hey Jerry,"** ask a question or give a command, and get a spoken answer in
about a second — with a follow-up window so you don't repeat the wake word.

Built as a systems-integration project: wake-word detection, neural voice
activity detection, streaming speech-to-text, a tiered "brain," and streamed
text-to-speech, all wired together on constrained hardware and tuned for how a
real conversation actually flows.

## Pipeline
```
wake ("hey jerry")            openWakeWord
  → record until endpoint     Silero VAD (neural, adaptive silence hang)
  → speech-to-text            Deepgram streaming  →  faster-whisper fallback (offline)
  → route                     instant command  |  fast tier (Haiku)  |  deep tier (agent + tools/memory)
  → text-to-speech            Piper (warm model, sentence-streamed)
  → speaker                   ReSpeaker mic array / 3.5mm out
```

## What makes it responsive (the interesting parts)
- **Adaptive endpointing** — silence-hang flexes on the *last word* of the live
  partial transcript: end fast on "yeah"/"ok", give grace on trailing "…and/…because".
- **Speculative dual dispatch** — for questions that *look* deep, the fast and
  deep tiers fire in parallel, so there's no serial "fast-decides-then-escalates" penalty.
- **Sentence-streamed TTS + barge-in** — playback starts as soon as sentence 1 is
  synthesized; sustained speech over the assistant stops it (guarded by high-confidence VAD).
- **Half-duplex echo control** — the mic is muted during playback so the assistant
  never hears itself off the speaker.
- **Instant local intents** — time/date/"turn on my PC" answered in <1s with no model call.
- **Self-healing** — a dropped USB mic triggers a PortAudio re-scan and restart instead of a crash.

## Tiered "brain"
1. **Instant** — local regex intents (time, date, Wake-on-LAN the PC).
2. **Fast** — direct Anthropic API (Haiku) for one-breath answers; replies `ESCALATE` when it can't.
3. **Deep** — the full agent via a gateway (tools + memory) for live info / multi-step work.

## Hardware
- Raspberry Pi + **ReSpeaker XVF3800** mic array (hardware AEC/beamforming)
- 3.5mm line-out to a powered speaker

## Files
- `jerry_voice.py` — the full agent (wake loop, VAD, STT, routing, TTS)
- `say.sh` — speak on the room speaker or the PC (Windows SAPI, Wake-on-LAN if asleep)
- `status.sh` — health check (room ready / listening / PC awake)

## Setup
```bash
cp jarvis.env.example jarvis.env          # fill in your hosts/keys (gitignored)
cp config.example.yaml ~/jarvis/config.yaml
bash scripts/install-hooks.sh             # enable the secret-scanning pre-commit hook
python3 jerry_voice.py --say "Hello"      # TTS test
python3 jerry_voice.py                     # run the wake loop
```
API keys are read from external files (`~/jarvis/.anthropic_key`, `.deepgram_key`)
or environment variables — never hardcoded.

## Security
No secrets or private network details live in this repo. Real hosts, keys, and
MAC addresses go in `jarvis.env` / `config.yaml` (both gitignored). A
**pre-commit hook** (`.githooks/pre-commit`) blocks any commit containing a
private IP, MAC address, or key — run `scripts/install-hooks.sh` once to enable it.

---
*Personal project. "Jerry" is the in-room voice front-end to a larger home
automation/agent system.*
