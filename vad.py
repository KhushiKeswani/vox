from silero_vad import load_silero_vad
import torch
import time
import numpy as np
from collections import deque
from config import PRE_BUFFER_CHUNKS,SAMPLE_RATE,VAD_THRESHOLD,SILENCE_CHUNKS
class VADSegmenter:
    def __init__(self):
        self.model = load_silero_vad()

        self.pre_buffer = deque(maxlen=PRE_BUFFER_CHUNKS)
        self.utterance_chunks = []
        self.speaking = False
        self.silence_count = 0
        self.speech_start_time = None
    def process_chunk(self, audio):
        self.pre_buffer.append(audio)

        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        speech_probability = self.model(audio_tensor, SAMPLE_RATE).item()

        if speech_probability > VAD_THRESHOLD:
            if not self.speaking:
                self.speaking = True
                self.speech_start_time = time.perf_counter()
                self.utterance_chunks.extend(list(self.pre_buffer))
            else:
                self.utterance_chunks.append(audio)

            self.silence_count = 0

        else:
            if self.speaking:
                self.utterance_chunks.append(audio)
                self.silence_count += 1

                if self.silence_count >= SILENCE_CHUNKS:
                    utterance = np.concatenate(self.utterance_chunks)

                    self.utterance_chunks.clear()
                    self.pre_buffer.clear()
                    self.speaking = False
                    self.silence_count = 0

                    return utterance

        return None