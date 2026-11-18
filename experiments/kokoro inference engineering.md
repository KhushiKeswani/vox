# Kokoro Inference Engineering

This document records experiments conducted to understand and optimize Kokoro TTS inference latency on a CPU-only development machine.

The goal was not to optimize for a specific benchmark number, but to identify where inference time was being spent and determine which optimizations were actually beneficial on constrained hardware.

---

## Hardware / Environment

* CPU: Intel Core i5-6300U
* CPU: 2 physical cores / 4 logical processors
* GPU: No dedicated GPU
* Inference: CPU
* Model: Kokoro TTS
* Primary metrics:

  * Time to first audio
  * Total Kokoro generation time
  * End-to-end turn latency where relevant

---

## 1. CPU Threading Experiment

### Question

Does increasing the number of CPU threads improve Kokoro inference?

### Experiment

Compared Kokoro execution using 1 thread and 4 threads.

### Results

| Configuration | First audio | Total Kokoro |
| ------------- | ----------: | -----------: |
| 1 thread      |     ~20.95s |      ~34.05s |
| 4 threads     |  **18.16s** |   **63.40s** |

### Observation

Increasing the number of threads reduced time to first audio in this run, but substantially increased total Kokoro latency.

This demonstrated that increasing CPU parallelism does not automatically improve end-to-end inference performance on constrained hardware.

---

## 2. End-to-End Stage Profiling

### Question

Where is time being spent during a complete voice-assistant turn?

### Measurement

One representative end-to-end turn was broken into pipeline stages.

| Stage                    |       Time |
| ------------------------ | ---------: |
| VAD                      |      1.54s |
| Whisper model loading    |     12.82s |
| Whisper transcription    |      6.17s |
| Router + Tavily + answer |      5.17s |
| Kokoro first audio       |     20.21s |
| Kokoro total             |     35.74s |
| **Total turn**           | **65.81s** |

### Observation

The measurement showed that Kokoro was a major contributor to the observed end-to-end latency, while Whisper model loading was also a significant cost.

This motivated a deeper investigation into Kokoro's internal inference stages.

---

## 3. Kokoro First-Audio / Total-Latency Measurement

### Question

How different are time-to-first-audio and total generation latency?

A later Kokoro run produced:

| Metric       |       Time |
| ------------ | ---------: |
| First audio  |  **6.98s** |
| Total Kokoro | **12.05s** |

Another run produced:

| Metric       |       Time |
| ------------ | ---------: |
| First audio  | **12.32s** |
| Total Kokoro | **23.39s** |

### Observation

Kokoro latency varied significantly between runs on the same CPU-only environment.

This made it important to distinguish:

* **TTFA:** how quickly the user receives the first audio
* **Total inference time:** how long the complete audio generation takes

For an interactive voice system, both metrics matter.

---

## 4. Kokoro Model Architecture Investigation

### Question

Is Kokoro a single monolithic computation, or can its inference stages be inspected independently?

Inspection of the pipeline exposed:

```text
repo_id
lang_code
model
voices
g2p
```

The underlying neural network was accessible through:

```text
pipeline.model
```

The model structure revealed multiple components including:

* BERT / ALBERT-based text processing
* Linear layers
* LSTM layers
* Conv1d layers
* ConvTranspose1d layers
* Text encoder
* Predictor
* F0 / noise-related prediction
* Decoder / generator

### Observation

The architecture could therefore be investigated at the component/operator level rather than treating Kokoro as a black box.

---

## 5. FP32 Component Profiling

### Question

Which major Kokoro components consume the most inference time?

### Results

| Component              |     Time |
| ---------------------- | -------: |
| BERT                   | **1.8s** |
| Predictor Text Encoder | **0.3s** |
| Predictor LSTM         | **0.1s** |
| F0/N Predictor         | **0.5s** |
| Text Encoder           | **0.8s** |
| Decoder                | **3.2s** |
| **Total**              | **6.7s** |

### Observation

The decoder was one of the largest measured components at **3.2s**.

The Predictor LSTM was comparatively inexpensive at **0.1s**.

This helped narrow the optimization target away from simply assuming that the LSTM was the primary bottleneck.

---

## 6. FP32 Operator Profiling

### Question

Which low-level operators dominate Kokoro inference?

Profiling identified convolution-heavy computation as a significant source of execution time.

Representative measurement:

```text
Convolution / Conv1d: ~3.26s
LSTM:                 ~0.23s
```

### Observation

The operator-level profile supported the component-level result: convolution/decoder-related computation was substantially more expensive than the LSTM portion in the measured run.

This demonstrated the value of profiling below the high-level module level.

---

## 7. INT8 Quantization Experiment

### Question

Can INT8 quantization reduce Kokoro CPU inference latency?

### Experiment

An INT8 quantization configuration was tested against FP32.

### Results

Representative profiler measurements:

| Configuration |       Time |
| ------------- | ---------: |
| FP32          |  **7.14s** |
| INT8          | **19.44s** |

The INT8 profile also showed:

```text
aten::_int_mm ≈ 12.31s
```

Other measured INT8 operations included approximately:

```text
Conv1d       3.62s
weight_norm  0.57s
Linear       0.42s
LSTM         0.10s
```

### Observation

INT8 was substantially slower in the tested configuration.

The profile exposed a large amount of time spent in the INT8 matrix multiplication path:

```text
aten::_int_mm
```

### Conclusion

Quantization was therefore **not adopted as an optimization** for this Kokoro deployment configuration.

The experiment demonstrated that:

> Quantization does not inherently make CPU inference faster. Its benefit depends on the model workload, hardware, and availability of efficient backend kernels.

---

## 8. FP32 vs INT8 Benchmarking

A separate timing comparison confirmed that the tested INT8 configuration did not provide the expected speedup.

The important engineering decision was therefore based on measurement rather than the assumption that:

```text
INT8 = faster
```

Instead:

```text
FP32 → benchmark
INT8 → benchmark
profile → identify regression
reject optimization
```

---

## 9. Warm / Cold Inference Investigation

### Question

How much of the apparent latency comes from model initialization versus actual inference?

The investigation separated:

```text
model loading
+
inference
```

rather than treating both as one TTS operation.

### Observation

Model loading can significantly affect the first request, while subsequent inference represents a different performance characteristic.

This reinforced the importance of keeping model initialization separate from steady-state inference measurements.

---

## 10. Streaming / Time-to-First-Audio Investigation

Kokoro output was investigated using first-audio timing rather than only measuring completion.

The conceptual distinction was:

```text
TTFA
│
├── preprocessing
├── initial inference
└── first audio available
     
Total
│
├── first audio
└── remaining generation
```

### Observation

Streaming audio can improve **perceived latency** because playback can begin before the complete response has been generated.

However:

> Streaming does not necessarily reduce the amount of computation required to generate the complete response.

This distinction became important when evaluating the voice assistant as an interactive system.

---

## 11. CPU Inference Investigation

Kokoro was investigated specifically under the project's deployment constraint:

```text
CPU-only
2 physical cores
4 logical processors
no dedicated GPU
```

The experiments showed that inference performance was highly dependent on CPU execution characteristics.

This led to treating CPU utilization, thread configuration, operator implementation, and model architecture as separate optimization dimensions.

---

# Key Findings

### 1. More threads were not automatically better

The 1-thread / 4-thread experiment produced different behavior for TTFA and total latency.

Therefore thread count needs to be benchmarked rather than selected purely from the number of logical CPUs.

### 2. The LSTM was not the main bottleneck

Measured LSTM time was approximately:

```text
0.1–0.23s
```

while decoder/convolution-related computation was several seconds.

### 3. Decoder/convolution computation was significant

Measured values included:

```text
Decoder:      3.2s
Conv1d:       ~3.26s
```

in representative profiling runs.

### 4. INT8 did not help this workload

The tested INT8 configuration increased measured inference time:

```text
FP32: 7.14s
INT8: 19.44s
```

with `aten::_int_mm` accounting for approximately **12.31s** in the INT8 profile.

### 5. TTFA and total latency are different metrics

A TTS system can start producing audio earlier without reducing total generation time.

For an interactive voice assistant, reducing TTFA can therefore be valuable even when total computation remains similar.

### 6. CPU inference has significant run-to-run variability

Kokoro measurements included:

```text
6.98s  first audio / 12.05s total

12.32s first audio / 23.39s total
```

This makes repeated measurements under controlled conditions important.

---

# Engineering Process

The Kokoro investigation followed this general loop:

```text
Observe latency
      ↓
Measure individual stages
      ↓
Inspect model architecture
      ↓
Profile operators/components
      ↓
Form optimization hypothesis
      ↓
Test one optimization
      ↓
Benchmark again
      ↓
Keep or reject based on measurements
```

The most important outcome was not a single optimization.

It was establishing a **measurement-driven approach to inference engineering** on constrained CPU hardware.

---

# Limitations

These measurements were collected during development on a specific CPU-only machine.

Results should not be interpreted as universal Kokoro performance characteristics.

The measurements were also collected under different experimental conditions, so numbers from separate experiments should not be directly compared unless the following were controlled:

* same text
* same voice
* same synthesis settings
* same CPU configuration
* same software environment
* same measurement method
* same warm/cold state

The reported numbers should therefore be treated as **experiment-specific observations**, not general benchmarks for Kokoro.

---

# Final Decision

Kokoro was investigated extensively as a potential TTS backend.

The experiments showed that further optimization required increasingly low-level investigation, while the tested INT8 approach did not improve performance on the target CPU.

The project therefore moved toward a lighter TTS backend better suited to the constrained deployment environment.

The Kokoro investigation was retained as an inference-engineering experiment because it demonstrated:

* performance measurement
* stage-level profiling
* model architecture inspection
* operator-level profiling
* CPU thread experimentation
* quantization evaluation
* TTFA vs total-latency analysis
* measurement-driven optimization decisions