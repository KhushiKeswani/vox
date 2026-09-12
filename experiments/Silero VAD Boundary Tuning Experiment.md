# Silero VAD Boundary Tuning Experiment

## Objective

Evaluate and tune Silero VAD temporal parameters for speech-segmentation accuracy.

The experiment focuses on how closely the detected speech start and end boundaries match manually annotated ground-truth boundaries.

---

## Experimental setup

**VAD:** Silero VAD v6.2.1
**VAD threshold:** 0.5
**Audio:** 10 manually annotated recordings
**Audio format:** 16 kHz mono

Each recording was manually inspected to establish the approximate ground-truth speech start and speech end times.

The VAD configuration was then evaluated against these annotations using boundary errors.

---

# 1. Ground-truth annotations

| Recording            | Speech start | Speech end |
| -------------------- | -----------: | ---------: |
| `Recording.wav`      |      0.790 s |    4.046 s |
| `Recording (10).wav` |      0.938 s |    6.304 s |
| `Recording (2).wav`  |      1.108 s |    4.027 s |
| `Recording (3).wav`  |      1.128 s |    3.770 s |
| `Recording (4).wav`  |      1.007 s |    5.552 s |
| `Recording (5).wav`  |      0.643 s |    4.730 s |
| `Recording (6).wav`  |      0.779 s |    3.369 s |
| `Recording (7).wav`  |      1.048 s |    6.310 s |
| `Recording (8).wav`  |      0.993 s |    6.319 s |
| `Recording (9).wav`  |      0.511 s |    6.950 s |

---

# 2. Parameter experiments

The experiment varied two temporal parameters:

* **Pre-buffer:** number of audio chunks retained before VAD detects speech.
* **Trailing silence:** number of consecutive silent chunks required before ending an utterance.

Each chunk represents approximately **32 ms** of audio.

|   Pre-buffer | Trailing silence | Mean absolute start error | Mean absolute end error |
| -----------: | ---------------: | ------------------------: | ----------------------: |
|    10 chunks |        19 chunks |                   ~276 ms |                 ~735 ms |
|     5 chunks |        19 chunks |                   ~114 ms |                 ~735 ms |
|     2 chunks |        19 chunks |                    ~19 ms |                 ~735 ms |
|    10 chunks |        10 chunks |                   ~276 ms |                 ~446 ms |
| **2 chunks** |    **10 chunks** |                **~19 ms** |             **~446 ms** |

---

# 3. Final V1 configuration

Based on the experiments, the following configuration was frozen for V1:

```text
Pre-buffer:       2 chunks ≈ 64 ms
Trailing silence: 10 chunks ≈ 320 ms
VAD threshold:    0.5
```

The configuration was chosen because it produced substantially better start-boundary accuracy while also reducing the end-boundary error compared with the original 10/19 configuration.

---

# 4. Final 2/10 experiment

For the final configuration, the measured start-boundary differences across the 10 recordings were:

```text
+10 ms
+20 ms
 +8 ms
+47 ms
 +3 ms
+11 ms
+24 ms
 +1 ms
+31 ms
+22 ms
```

Mean absolute start-boundary error:

```text
≈ 19 ms
```

The positive values indicate that the detected boundary occurred slightly before the manually annotated speech boundary under the error convention used in this experiment.

---

## End-boundary errors

The corresponding end-boundary differences were:

```text
-448 ms
-453 ms
-486 ms
-432 ms
-486 ms
-471 ms
-442 ms
-497 ms
-314 ms
-434 ms
```

Mean absolute end-boundary error:

```text
≈ 446 ms
```

---

# 5. Manual validation

All 10 recordings were visually checked after applying the final 2/10 configuration.

No premature speech cutoffs were observed during this validation.

The approximately 446 ms end-boundary error therefore represents **late stopping**, rather than premature truncation, under the error convention used here.

---

# 6. Result

The parameter sweep showed that the pre-buffer primarily affected speech-start detection, while trailing-silence duration strongly affected the end boundary.

The final configuration reduced mean absolute start-boundary error from approximately:

```text
276 ms → 19 ms
```

while reducing mean absolute end-boundary error from:

```text
735 ms → 446 ms
```

The 2/10 configuration was therefore frozen as the V1 VAD configuration.

---

# 7. Evidence

This experiment is based on:

* 10 manually annotated recordings
* Multiple tested VAD parameter configurations
* Per-recording boundary errors
* Mean absolute boundary-error calculations
* Manual visual validation of the final configuration

The experiment was performed during development of the CPU-only voice assistant.

---

# 8. Reproducibility

To reproduce the experiment:

1. Use the same 10 recordings.
2. Use Silero VAD with threshold `0.5`.
3. Evaluate the tested pre-buffer/trailing-silence combinations.
4. Compare detected speech boundaries against the ground-truth annotations above.
5. Calculate mean absolute start and end boundary errors.
6. Visually inspect the final segmentation for premature cutoffs.
7. Freeze the configuration based on the measured results.
