
# Vox

A CPU-only, streaming voice assistant. Mic in, voice out, running on a dual-core laptop with no GPU.

```
Microphone → Silero VAD → Whisper (batch) → Gemini (streaming) → Piper TTS → Speaker
```

The hard part isn't connecting four models. It's making the audio loop behave like a conversation: detect when the user starts and stops, transcribe what they said, generate a response, and start speaking before the response is fully generated.

---

## Results

### Speech-boundary detection

Measured across 10 manually annotated recordings.

| Configuration | Start error (mean abs) | End error (mean abs) | Premature cutoffs |
|---|---|---|---|
| Original (pre-buffer 10, silence 19) | ~276 ms | ~735 ms | — |
| Tuned (pre-buffer 2, silence 10) | **~19 ms** | ~446 ms | **0** |

**~93.1% reduction in start-boundary error.** End error is still ~446 ms, which is the right next thing to fix.

### Gemini streaming interface

Same prompts, two different streaming interfaces.

| Interface | First-chunk latency |
|---|---|
| Interactions API | 18.7 s average |
| Regular streaming | 0.89 – 1.36 s on 7 of 10 runs |

Not a matched percentile comparison, but a clear signal to use regular streaming. Rough first-chunk latency drop is ~93%.

### End-to-end latency

100 turns profiled. Total turn latency includes user speaking time, since the timer starts before mic capture.

| Stage | Median | P99 |
|---|---|---|
| End-to-end turn | 12.5 s (P50), 16.0 s (P95), 17.8 s (P99) | — |
| Whisper | 1.6 s | 9.4 s (~5.9× median) |
| Piper | 3.3 s | 10.9 s (~3.3× median) |
| Gemini | — | 3.8 s |

The tail is in local CPU inference, not the LLM. Whisper and Piper P99 spikes drive the long turns. See the *Hardware caveat* below for why these numbers are shaped the way they are.

---

## Architecture

```
┌──────────┐      ┌──────────┐      ┌──────────┐      ┌─────────────────┐      ┌──────────┐      ┌─────────┐
│   Mic    │ ───► │  Silero  │ ───► │ Whisper  │ ───► │ Gemini (stream) │ ───► │  Piper   │ ───► │ Speaker │
│ 16 kHz   │      │   VAD    │      │  (batch) │      │                 │      │ (worker) │      │         │
└──────────┘      └──────────┘      └──────────┘      └─────────────────┘      └──────────┘      └─────────┘
     │                 │                 │                    │                      ▲
     │  32 ms chunks   │  utterance      │  transcript        │  sentence buffer     │
     ▼                 ▼                 ▼                    ▼                      │
 audio queue      pre-buffer +      language="en"       "." "?" "!" splits    tts_queue (thread-safe)
                  silence counter                                                    ▲
                                                                                     │
                                                                              producer-consumer
```

**Two concurrency decisions make it feel live:**

1. **Audio callback is minimal.** It copies samples and puts them on a queue. Nothing else. All inference runs on other threads so capture can never stall.
2. **Producer-consumer between Gemini and Piper.** Gemini streams tokens, I buffer them into sentences, push to `tts_queue`. A Piper worker thread pulls sentences and plays them while Gemini is still generating. No waiting for the full response before TTS starts.

---

## The pipeline, stage by stage

### 1. Audio capture (`audio.py`)

| Setting | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Channels | 1 (mono) |
| Data type | float32 |
| Chunk size | 512 samples = **32 ms** |

Why 16 kHz: Whisper and Silero VAD standard. Enough resolution for speech, no wasted high frequencies.
Why mono: one mic, one voice. Stereo is doubled data for no benefit.
Why 512-sample chunks: 32 ms is a manageable VAD interval. Small enough to be responsive, big enough to run inference on.

The `sounddevice.InputStream` callback copies one channel into a queue. The main loop consumes it on a different thread. If the callback itself ran inference, the sound card would miss subsequent chunks.

### 2. Voice Activity Detection (`vad.py`)

Silero VAD returns a speech probability per chunk. Threshold = 0.5.

Four pieces of state:

| Variable | Purpose |
|---|---|
| `pre_buffer` | Rolling deque of recent chunks, kept before speech is detected |
| `utterance_chunks` | Current speech segment being collected |
| `speaking` | Whether an utterance is active |
| `silence_count` | Consecutive non-speech chunks since speech ended |

**Start problem.** VAD is reactive. It can only classify a chunk after receiving it. By then the first syllable is in the past.

**Fix:** 2-chunk pre-buffer (64 ms of history). When speech is first detected, prepend buffered chunks to the utterance. Whisper gets the onset, not a clipped word.

**End problem.** Speech has pauses, breaths, hesitations. Ending on the first silent chunk cuts the user off.

**Fix:** wait 10 silence chunks (= 320 ms) before closing the utterance. Below 10 I saw premature cutoffs.

### 3. Whisper (`stt.py`)

Model: `base`. Loaded once at startup.

Why not `small`: `small` had better accuracy but longer loading and transcription times. For this use case the loss was acceptable.

Why not `faster-whisper` INT8: tested it. Transcription quality regressed. Faster isn't useful if it's wrong.

Why `language="en"`: unconstrained detection sometimes produced incorrect-language output. Removing the uncertainty was free.

ASR is **batch, not streaming.** It runs on the completed utterance from VAD.

### 4. Gemini (`llm.py`)

Uses `generate_content_stream`, so text arrives incrementally.

System instruction: conversational English, one or two sentences. Max output: 25 tokens. Minimal thinking.

Why short: every token has to be synthesized and played. Longer responses mean longer waits.

**Interface comparison** (same prompts):

| Interface | First-chunk latency |
|---|---|
| Interactions API | 18.7 s average |
| Regular streaming | 0.89 – 1.36 s on 7 of 10 runs |

Regular streaming won by ~93%. Choosing the right interface mattered more than any prompt tweak.

### 5. Piper TTS (`tts.py`)

Local, CPU TTS. Model loaded once.

The important part is the producer-consumer design:

```python
# Producer (main thread, in voice_pipeline.py)
sentence_buffer = ""
for chunk in gemini_stream:
    sentence_buffer += chunk
    if sentence_buffer.endswith(('.', '?', '!')):
        tts_queue.put(sentence_buffer)
        sentence_buffer = ""

# Consumer (worker thread, in tts.py)
while True:
    text = tts_queue.get()
    if text is None:       # shutdown sentinel
        break
    audio_chunks = piper.synthesize(text)
    stream.write(audio_chunks)
```

**Sentence-level chunking, not token-level.** Piper gets whole sentences, because word-by-word synthesis sounds choppy and has no prosody. Sentences are the smallest unit that still sounds human.

Without overlap, total turn time = Gemini time + Piper time. With overlap, time to first audio drops to (Gemini first sentence + Piper first sentence), and the rest streams on top.

---

## How I measured 276 → 19 ms

1. Recorded 10 utterances on the same mic.
2. Manually annotated ground-truth start and end timestamps per recording.
3. Ran VAD with the original config (pre-buffer 10, silence 19). Mean absolute start error: ~276 ms. End error: ~735 ms.
4. Tuned to pre-buffer 2, silence 10. Mean absolute start error: ~19 ms. End error: ~446 ms. Zero premature cutoffs.

**Honesty notes:**
- Start boundary only. Not overall boundary error.
- Both variables changed together, so this isn't a clean ablation. Each one's contribution is next on the list to measure.
- 10 recordings is a small benchmark. Enough to tune, not enough to claim production quality.

---

## Profiling 100 turns

`voice_pipeline.py` logs per-stage timings with `time.perf_counter()`:

- Listening/VAD duration
- Total VAD compute time
- Whisper transcription time
- Time to first LLM chunk
- LLM answer generation time
- Time to first audio (speech end → Piper first chunk)
- Total turn latency

The P99-vs-median gap on Whisper and Piper (~5.9× and ~3.3×) pointed me at local CPU contention, not the LLM.

---

## Hardware caveat

Everything above runs on a Dell Latitude E5470: Intel i5-6300U, 2 physical cores, 4 GB RAM, no GPU, Windows 10.

The latency tails here are hardware-bound, not algorithm-bound:
- Whisper P99 of 9.4 s on CPU is roughly sub-second on a T4.
- Piper P99 of 10.9 s is the same story. Local CPU TTS is slow and jittery under load.
- Gemini interface choice matters a lot on the free tier, as the 18.7 s → 0.89 s comparison shows.

The numbers I controlled (speech-boundary error, interface choice, concurrency design) are what the project is about. The rest are a hardware ceiling, not a design ceiling.

---

## Engineering decisions and lessons

| Decision | Why |
|---|---|
| Keep audio callback to a copy + queue push | Blocking inside the callback drops audio chunks |
| Load models at startup | Reloading per turn is unusable |
| Whisper `base`, not `small` or INT8 | Measured both. Base won on CPU |
| Force `language="en"` | Removes random language-detection failures |
| Sentence-level TTS chunking (not token-level) | Prosody. Words alone sound robotic |
| Producer-consumer between LLM and TTS | Overlap generation with synthesis |
| 2-chunk pre-buffer | Preserves the onset without anchoring the start timestamp too far back |
| 10-chunk silence end-of-speech | Survives normal pauses, zero premature cutoffs |
| Regular streaming over Interactions API | ~93% lower first-chunk latency on same prompts |

---

## Known gaps (would fix next)

- **`piper_done` event never cleared.** After turn 1 it stays set, so `finish()` returns immediately on turn 2. Fix: clear at the top of `start()` and join the worker.
- **Gemini `ClientError` skips worker shutdown.** Wrap the generation loop in `try/finally` so cleanup is unconditional.
- **TTS queue is unbounded.** Needs backpressure. If Gemini outpaces Piper, memory grows.
- **End-boundary error (446 ms) is too high for code-switched speech.** A per-language silence threshold trained on real call recordings would help.
- **Not full-duplex.** User can't interrupt during playback.
- **Variables tuned together.** Ablation (change one at a time) is next.
- **Benchmark size is 10 recordings.** Enough to tune, not enough to generalize.

---

## Stack

- Python 3.10
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [openai-whisper](https://github.com/openai/whisper) (model: `base`)
- [google-generativeai](https://github.com/google/generative-ai-python) (streaming)
- [Piper TTS](https://github.com/rhasspy/piper)
- `sounddevice`, `numpy`, `torch` (CPU-only)

## Run it

```bash
pip install -r requirements.txt
# set GEMINI_API_KEY in .env
python voice_pipeline.py
```

---

## Why I built this

To understand voice AI as a systems and inference problem, not just an LLM API wrapper. Every stage here was a choice between latency, accuracy, and cost, measured on a 4 GB CPU-only laptop. The code is honest about what I controlled and what I didn't.
