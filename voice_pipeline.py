import time
import sounddevice as sd
import numpy as np
from collections import deque
import torch
import whisper
from faster_whisper import WhisperModel
import queue
import os
import threading
from piper import PiperVoice
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from silero_vad import load_silero_vad
startup_start = time.perf_counter()
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
print("PyTorch threads:", torch.get_num_threads())
print("PyTorch interop threads:", torch.get_num_interop_threads())
print("Loading Silero VAD...")
vad_start = time.perf_counter()
model = load_silero_vad()
print(f"Silero VAD loaded in {time.perf_counter() - vad_start:.2f}s")
MODEL = "piper/en_US-lessac-medium.onnx"
voice = PiperVoice.load(MODEL)

print(f"Startup complete in {time.perf_counter() - startup_start:.2f}s")

whisper_model = None
if whisper_model is None:
        print("Loading Whisper...")
        whisper_start = time.perf_counter()
        whisper_model = whisper.load_model("base")
        print("Whisper loaded.")
        print(
            f"Whisper loaded in "
            f"{time.perf_counter() - whisper_start:.2f}s")
        
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
print(api_key is None)
audio_queue = queue.Queue()
SAMPLE_RATE = 16000
CHUNK_SIZE = 512

system_instruction="""You are a conversational voice assistant.

Your responses will be converted directly into speech using TTS.

Keep every response concise:
-if ans can be given in 1 sentence keep it like that, if user asked is being passed then 1 sentence works.
- Prefer 1–2 sentences.
- Usually stay under 40 words.
- For simple questions, answer in one sentence.
- Only give a longer explanation when the user explicitly asks for detail.
-maximum 2 sentence
Speak naturally:
- Use conversational English.
- Use contractions like "I'm", "that's", "it's", "you'll".
- Avoid essay-like or overly formal language.
- No markdown, lists, headings, or unnecessary formatting.
- Don't repeat the user's question.
- Get straight to the answer."""
import time
def callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0].copy()
    audio_queue.put(audio)
def piper_worker():
        global first_audio_time

        with sd.OutputStream(
            samplerate=22050,
            channels=1,
            dtype="float32"
        ) as audio_stream:

            while True:
                text = tts_queue.get()

                if text is None:
                    tts_queue.task_done()
                    break
                try:
                    for chunk in voice.synthesize(text):

                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()

                            print(
                                f"Time to first audio: "
                                f"{first_audio_time - speech_end_time:.2f}s"
                            )

                        audio_stream.write(chunk.audio_float_array)
                except Exception as e:
                     print(f'Piper text to speech conversion error: {e}')

                tts_queue.task_done()

        piper_done.set()
utterance_chunks = []
utterance_ready = False
silence_count = 0
speech_count = 0
speaking = False
pre_buffer = deque(maxlen=2)
processed = False

print("Listening...")
while True:
    turn_start = time.perf_counter()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=callback
        ):
            vad_total_time = 0

            while not utterance_ready:
                audio = audio_queue.get()
                pre_buffer.append(audio)
                audio_tensor = torch.from_numpy(audio.astype(np.float32))

                vad_start = time.perf_counter()
                speech_probability = model(audio_tensor, SAMPLE_RATE).item()
                vad_total_time += time.perf_counter() - vad_start

                if speech_probability > 0.5:
                    if not speaking:
                        speaking = True
                        speech_start_time = time.perf_counter()
                        print("speech started")
                        utterance_chunks.extend(list(pre_buffer))
                    else:
                        utterance_chunks.append(audio)

                    silence_count = 0
                    speech_count += 1
                else:
                    if speaking == True:
                        utterance_chunks.append(audio)
                        silence_count += 1
                        speech_count = 0

                        if silence_count >= 10:
                            print("speech finished")
                            speech_end_time = time.perf_counter()
                            utterance_ready = True
                            break

        print(
            f"Listening/VAD duration: "
            f"{speech_end_time - speech_start_time:.2f}s"
        )

        print(f"Total VAD compute time: {vad_total_time:.2f}s")

        chunkres = np.concatenate(utterance_chunks)
        speaking = False
        try:
            whisper_start = time.perf_counter()
            transcription = whisper_model.transcribe(
            chunkres,
            language="en")
            final_text = transcription["text"].strip()
            print(final_text)
            print(
                        f"Whisper transcription took "
                        f"{time.perf_counter() - whisper_start:.2f}s"
                    )
        except Exception as e:
            print(f'Whisper Transcription error: {e}')
            continue

        print(final_text)

        if 'bye' in final_text.lower():
            break
        tts_queue = queue.Queue()

        first_audio_time = None
        piper_done = threading.Event()


            # Start Piper worker
        piper_thread = threading.Thread(
                target=piper_worker
            )

        piper_thread.start()
        llm_start = time.perf_counter()
        GEMINI_MODEL = "gemini-3.5-flash-lite"
        try:
            response = client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=final_text,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="minimal"
                        ),
                        max_output_tokens=25,
                    ),
                )
            text_chunks = []
            sentence_buffer = ""
            
            first_chunk_latency = None
            
            for chunk in response:
                            if chunk.text:
            
                                if first_chunk_latency is None:
                                    first_chunk_latency = time.perf_counter() - llm_start
            
                                    print(
                                        f"Time to first LLM chunk: "
                                        f"{first_chunk_latency:.2f}s"
                                    )
            
                                text = chunk.text
            
                                text_chunks.append(text)
                                sentence_buffer += text
            
                                if sentence_buffer.endswith((".", "?", "!")):
                                    tts_queue.put(sentence_buffer)
                                    sentence_buffer = ""
            if sentence_buffer.strip():
                            tts_queue.put(sentence_buffer)
        except ClientError as e:
            print(f"Gemini error: {e}")
            tts_queue.put(None)
            piper_done.wait()
            utterance_chunks.clear()

            while not audio_queue.empty():
                audio_queue.get_nowait()

            utterance_ready = False
            pre_buffer.clear()

            continue
        llm_latency = time.perf_counter() - llm_start

        answer = "".join(text_chunks)

        print(answer)

        print(
                f"LLM answer generation took "
                f"{llm_latency:.2f}s"
            )
        tts_queue.put(None)

        # Wait for Piper to finish speaking
        piper_done.wait()


        print(
                f"TOTAL TURN LATENCY: "
                f"{time.perf_counter() - turn_start:.2f}s"
            )

        utterance_chunks.clear()

        while not audio_queue.empty():
                audio_queue.get_nowait()
        utterance_ready = False
        pre_buffer.clear()
    except Exception as e:
            print(f'error while taking audio input: {e}')