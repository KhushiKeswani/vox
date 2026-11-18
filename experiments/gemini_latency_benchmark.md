# Gemini Latency Benchmark

## Objective

Compare Gemini's **Interactions API** against the standard streaming API for first-token latency in the voice assistant.

The goal was to determine which interface was more suitable for a latency-sensitive voice interaction pipeline.

---

## Experiment setup

**Model:** `gemini-3.5-flash-lite`
**Thinking:** `minimal`
**Prompt:** `"Hello"`
**Runs per interface:** 10
**Metric:** Time to First Token (TTFT)

TTFT measures the elapsed time between sending the request and receiving the first generated text token.

These measurements were collected locally during development of the voice assistant.

---

# Experiment 1 — Gemini Interactions API

The Interactions API produced the following measurements:

| Run | TTFT (s) | Total (s) |
| --: | -------: | --------: |
|   1 |    25.36 |     28.99 |
|   2 |    20.45 |     24.58 |
|   3 |    27.11 |     30.33 |
|   4 |    19.54 |     23.69 |
|   5 |    24.85 |     28.79 |
|   6 |     1.92 |      4.52 |
|   7 |    24.62 |     27.81 |
|   8 |     2.23 |      4.99 |
|   9 |    20.38 |     24.84 |
|  10 |    20.67 |     23.72 |

### TTFT observations

* Mean TTFT: **~18.71 s**
* Fast runs: **2/10**
* Fast-run TTFT: **1.92–2.23 s**
* Remaining runs: **~19.54–27.11 s**

The Interactions API showed a highly variable TTFT distribution, with most runs taking approximately 20–27 seconds.

---

# Experiment 2 — Standard Streaming API

The standard streaming API produced:

| Run | TTFT (s) | Total (s) |
| --: | -------: | --------: |
|   1 |     1.33 |      1.36 |
|   2 |    23.65 |     23.69 |
|   3 |     1.19 |      1.21 |
|   4 |     0.92 |      0.95 |
|   5 |     1.29 |      1.29 |
|   6 |     1.02 |      1.03 |
|   7 |     0.89 |      0.89 |
|   8 |    20.71 |     20.73 |
|   9 |    20.85 |     20.86 |
|  10 |    20.92 |     20.92 |

### TTFT observations

* Fast runs: **7/10**
* Fast-run TTFT: **0.89–1.33 s**
* Slow runs: **3/10**
* Slow-run TTFT: **20.71–23.65 s**

The standard streaming API also exhibited a long tail, but most runs returned the first token in approximately one second.

---

# Comparison

| Metric                   | Interactions API | Standard Streaming |
| ------------------------ | ---------------: | -----------------: |
| Runs                     |               10 |                 10 |
| Fast runs                |             2/10 |               7/10 |
| Fast TTFT range          |      1.92–2.23 s |        0.89–1.33 s |
| Observed slow TTFT range |    19.54–27.11 s |      20.71–23.65 s |
| Mean TTFT                |         ~18.71 s |            ~9.28 s |

The standard streaming API had a substantially lower observed average TTFT and produced fast first-token responses in **7/10 runs**, compared with **2/10** for the Interactions API.

---

# Decision

For the V1 voice assistant, the **standard Gemini streaming API was selected** over the Interactions API.

The decision was based on measured latency rather than assuming that one interface would be faster from documentation or API design alone.

The experiment also revealed that the standard streaming API still had significant long-tail latency: **3/10 runs exceeded 20 seconds TTFT**.

Therefore, switching APIs improved the typical observed latency but did **not** eliminate the underlying long-tail behavior.

---

# Raw terminal evidence

## Interactions API

```text
Run 1: TTFT=25.36s | Total=28.99s
Run 2: TTFT=20.45s | Total=24.58s
Run 3: TTFT=27.11s | Total=30.33s
Run 4: TTFT=19.54s | Total=23.69s
Run 5: TTFT=24.85s | Total=28.79s
Run 6: TTFT=1.92s | Total=4.52s
Run 7: TTFT=24.62s | Total=27.81s
Run 8: TTFT=2.23s | Total=4.99s
Run 9: TTFT=20.38s | Total=24.84s
Run 10: TTFT=20.67s | Total=23.72s
```

## Standard streaming API

```text
Run 1: TTFT=1.33s | Total=1.36s
Run 2: TTFT=23.65s | Total=23.69s
Run 3: TTFT=1.19s | Total=1.21s
Run 4: TTFT=0.92s | Total=0.95s
Run 5: TTFT=1.29s | Total=1.29s
Run 6: TTFT=1.02s | Total=1.03s
Run 7: TTFT=0.89s | Total=0.89s
Run 8: TTFT=20.71s | Total=20.73s
Run 9: TTFT=20.85s | Total=20.86s
Run 10: TTFT=20.92s | Total=20.92s
```

---

## Limitations

This was a **development benchmark**, not a production-scale performance study.

Only 10 runs were performed for each interface, so the results should not be interpreted as a universal performance comparison between the two Gemini interfaces.

The purpose was to make an engineering decision for this specific voice assistant under the tested conditions.

---

## Reproducibility

To reproduce this experiment:

1. Use the same model: `gemini-3.5-flash-lite`
2. Use `minimal` thinking
3. Send the same `"Hello"` prompt
4. Run each interface 10 times
5. Record TTFT and total generation time
6. Compare the resulting distributions