import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

class HuggingFaceLLM:
    """
    Wrapper for HuggingFace Endpoint using LangChain-HuggingFace.
    Validates environment config and prevents runtime errors due to missing provider/model.
    """

    def __init__(self, temperature=0.3, max_new_tokens=512):
        load_dotenv()

        model_id = os.getenv("HF_MODEL_ID")
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        if not model_id or not api_token:
            raise ValueError("Missing HuggingFace config in .env file.")

        # Print debug info (optional)
        print(f"Using Model ID: {model_id}")
        print(f"API Token starts with: {api_token[:5]}...")

        # 🧠 Ensure model is compatible with text generation
        known_good_models = [
            "mistralai/Mistral-7B-Instruct-v0.1",
            "HuggingFaceH4/zephyr-7b-beta",
            "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-rw-1b",  # ✅ Added
    "google/flan-t5-base",  # ✅ Added
    "Writer/palmyra-base"   # ✅ Added
        ]
        if model_id not in known_good_models:
            print("⚠️ Warning: Model ID is not in known-good list. Make sure it's valid for text-generation!")

        # 🛠️ Instantiate the endpoint safely
        self.llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=api_token,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

    def generate(self, prompt: str) -> str:
        try:
            return self.llm.invoke(prompt)
        except StopIteration:
            return "❌ Error: No provider found. Double-check model ID and API token."
        except Exception as e:
            return f"🚨 Unexpected error occurred: {str(e)}"
