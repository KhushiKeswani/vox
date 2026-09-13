from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from config import GEMINI_MODEL,MAX_OUTPUT_TOKENS
import os
class LLM:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = genai.Client(api_key=self.api_key)
        self.system_instruction = """You are a conversational voice assistant.

Your responses will be converted directly into speech using TTS.

Keep every response concise:
-if ans can be given in 1 sentence keep it like that, if user asked is being passed then 1 sentence works.
- Prefer 1–2 sentences.
- Usually stay under 25 words.
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
    def chat(self,final_text):
        response = self.client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=final_text,
                        config=types.GenerateContentConfig(
                            system_instruction=self.system_instruction,
                            thinking_config=types.ThinkingConfig(
                                thinking_level="minimal"
                            ),
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        ),
                    )
        for chunk in response:
            if chunk.text:
                yield chunk.text