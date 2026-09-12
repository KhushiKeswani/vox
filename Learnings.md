1. Making an AI project handling real time audio is very different than traditional text based chatbots(learnt channels of audio, bandwidth, etc)
2. Never put expensive/blocking work inside the real-time audio callback.
Callbacks need to return quickly; heavy processing should happen outside them.
3. Whisper can produce incorrect-language output when the language is unconstrained.
For a known English application, explicitly setting language="en" made the ASR behavior appropriate for the use case.
4. VAD can miss the beginning of speech because detection happens after audio has already arrived.
A pre-buffer preserves recent chunks and allows them to be prepended when speech is detected.
5. API limits are an engineering constraint, not just a product limitation.
When Gemini's daily call limit became restrictive, I tested an OpenRouter alternative; it had much higher observed latency (~39s) and lacked current-information capability for my use case.
6. The fastest model isn't necessarily the model with the fewest parameters.
Inference runtime, computational graph, operators and hardware can have a major effect on actual latency.
7. Producer-consumer architecture is useful for AI pipelines.
Treating audio capture and processing as independent producers/consumers makes concurrency and bottlenecks easier to reason about.
8. Latency optimization in voice bot is a real engineering task. voicebot requires much lower latency than traditional chatbots.
9. Locally ran models(Whisper, Piper) learnt and experimented about Whisper model sized(ran experiment on tiny,base,small,turbo).initially chose small because of highest accuracy, but then lower it to base because of loading and transcribing latency overhead. compared base and small accuracy, and resulted as loss of accuracy was balanced with llm's assumption of words.
10. Experimented with faster-whisper, Tested faster-whisper + INT8, tested it and rejected it because the transcription quality was worse for our use case.
11. For highest utilization, worked on cpu thread level and utilized all 4 threads of CPU.
12. Differentiated system's latency with different component's latency and tried best to optimize it
13. Learnt alot about inference engineering(tbh I only thought about quantization model,but ended up profiling and analyzing which layer is occupying most time and quantized it, ended up with even bad results)