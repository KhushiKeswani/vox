import time
import torch
from vad import VADSegmenter
from dotenv import load_dotenv
load_dotenv()
from google.genai.errors import ClientError
from audio import AudioCapture
from stt import WhisperTranscriber
from llm import LLM
from tts import Tts

audio_capture = AudioCapture()

startup_start = time.perf_counter()

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

print("PyTorch threads:", torch.get_num_threads())
print("PyTorch interop threads:", torch.get_num_interop_threads())

vad = VADSegmenter()

voice = Tts()

print(f"Startup complete in {time.perf_counter() - startup_start:.2f}s")

print("Loading Whisper...")
whisper_start = time.perf_counter()
whisper_model = WhisperTranscriber()
print("Whisper loaded.")
print(
    f"Whisper loaded in "
    f"{time.perf_counter() - whisper_start:.2f}s"
)
llm = LLM()
print("Listening...")
turn_count = 0

while True:
    turn_start = time.perf_counter()
    while not audio_capture.audio_queue.empty():  
        audio_capture.audio_queue.get_nowait()
    try:
        with audio_capture.start():
            vad_total_time = 0

            while True:
                audio = audio_capture.audio_queue.get()

                vad_start = time.perf_counter()
                utterance = vad.process_chunk(audio)
                vad_total_time += time.perf_counter() - vad_start

                if utterance is not None:
                    speech_end_time = time.perf_counter()
                    speech_start_time = vad.speech_start_time
                    break

        print(
            f"Listening/VAD duration: "
            f"{speech_end_time - speech_start_time:.2f}s"
        )

        print(f"Total VAD compute time: {vad_total_time:.2f}s")

        chunkres = utterance

        try:
            whisper_start = time.perf_counter()

            final_text = whisper_model.transcribe(chunkres)

            print(
                f"Whisper transcription took "
                f"{time.perf_counter() - whisper_start:.2f}s"
            )

        except Exception as e:
            print(f"Whisper Transcription error: {e}")
            continue

        print(final_text)

        if "bye" in final_text.lower():
            break
        llm_start = time.perf_counter()

        try:

            text_chunks = []
            sentence_buffer = ""
            voice.start(speech_end_time)
            first_chunk_latency = None
            
            for chunk in llm.chat(final_text):
                    if first_chunk_latency is None:
                        first_chunk_latency = (
                            time.perf_counter() - llm_start
                        )

                        print(
                            f"Time to first LLM chunk: "
                            f"{first_chunk_latency:.2f}s"
                        )
                    text_chunks.append(chunk)
                    sentence_buffer += chunk

                    if sentence_buffer.endswith((".", "?", "!")):
                        voice.speak(sentence_buffer)
                        sentence_buffer = ""

            if sentence_buffer.strip():
                voice.speak(sentence_buffer)

        except ClientError as e:
            print(f"Gemini error: {e}")

            voice.speak(sentence_buffer)
            voice.finish()
            continue

        llm_latency = time.perf_counter() - llm_start

        answer = "".join(text_chunks)

        print(answer)

        print(
            f"LLM answer generation took "
            f"{llm_latency:.2f}s"
        )

        voice.finish()

        print(
            f"TOTAL TURN LATENCY: "
            f"{time.perf_counter() - turn_start:.2f}s"
        )

        turn_count += 1
        print(turn_count)

    except Exception as e:
        print(f"error while taking audio input: {e}")