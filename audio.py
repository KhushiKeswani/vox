import queue
import sounddevice as sd

from config import SAMPLE_RATE, CHUNK_SIZE


class AudioCapture:
    def __init__(self):
        self.audio_queue = queue.Queue()

    def callback(self, indata, frames, callback_time, status):
        if status:
            print(status)

        audio = indata[:, 0].copy()
        self.audio_queue.put(audio)

    def start(self):
        return sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=self.callback,
        )