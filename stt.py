import whisper
from config import WHISPER_MODEL


class WhisperTranscriber:
    def __init__(self):
        self.model = whisper.load_model(WHISPER_MODEL)

    def transcribe(self, audio):
        result = self.model.transcribe(
            audio,
            language="en"
        )
        return result["text"].strip()