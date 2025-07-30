import os
import google.generativeai as genai

class GeminiLLM:
    def __init__(self, temperature=0.3, model_name=None):
        api_key = os.getenv("GEMINI_API_KEY")
        model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "models/gemini-pro")

        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 1024
            }
        )
        return response.text.strip()
