from piper import PiperVoice
from config import PIPER_MODEL,PIPER_SAMPLE_RATE
import sounddevice as sd
import queue
import threading
import time
class Tts:
    def __init__(self):
        self.voice = PiperVoice.load(PIPER_MODEL)
        self.tts_queue = queue.Queue()
        self.first_audio_time = None
        self.speech_end_time = None
        self.piper_done = threading.Event()
    def piper_worker(self):
        try:
            with sd.OutputStream(
                samplerate=PIPER_SAMPLE_RATE,
                channels=1,
                dtype="float32"
            ) as audio_stream:

                while True:
                    text = self.tts_queue.get()

                    if text is None:
                        self.tts_queue.task_done()
                        break

                    try:
                        for chunk in self.voice.synthesize(text):

                            if self.first_audio_time is None:
                                self.first_audio_time = time.perf_counter()

                                print(
                                    f"Time to first audio: "
                                    f"{self.first_audio_time - self.speech_end_time:.2f}s"
                                )

                            audio_stream.write(chunk.audio_float_array)

                    except Exception as e:
                        print(f"Piper text to speech conversion error: {e}")

                    self.tts_queue.task_done()

        finally:
            self.piper_done.set()
    def start(self, speech_end_time):
        self.piper_done.clear()        
        self.first_audio_time = None
        self.speech_end_time = speech_end_time
        self.piper_thread = threading.Thread(target=self.piper_worker)
        self.piper_thread.start()
    def speak(self,text):
        self.tts_queue.put(text)
    def finish(self):
        self.tts_queue.put(None)
        self.piper_done.wait()