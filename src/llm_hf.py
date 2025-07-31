# import os
# import logging
# from typing import Optional
# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint

# # Load environment variables
# load_dotenv()

# # Configure logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# class HuggingFaceLLM:
#     """
#     Wrapper for Hugging Face models via LangChain's HuggingFaceEndpoint.
#     Uses conversational task only (for models like Kimi-K2-Instruct).
#     """

#     def __init__(
#         self,
#         model_name: Optional[str] = None,
#         temperature: float = 0.3,
#         max_tokens: int = 1024,
#         top_p: float = 0.9,
#         repetition_penalty: float = 1.1,
#         enable_cache: bool = False
#     ):
#         self.model_name = model_name or os.getenv("LLM_MODEL")
#         self.api_key = os.getenv("HUGGINGFACE_API_KEY")

#         if not self.model_name:
#             raise ValueError("❌ Model name not provided. Set `LLM_MODEL` in .env or pass explicitly.")
#         if not self.api_key:
#             raise ValueError("❌ Hugging Face API key not found. Set `HUGGINGFACE_API_KEY` in .env")

#         self.enable_cache = enable_cache
#         self.cache = {}

#         try:
#             self.llm = HuggingFaceEndpoint(
#                 repo_id=self.model_name,
#                 huggingfacehub_api_token=self.api_key,
#                 task="conversational",  # ✅ Force conversational task
#                 temperature=temperature,
#                 max_new_tokens=max_tokens,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 do_sample=True
#             )
#             logger.info(f"✅ Initialized HuggingFace LLM: {self.model_name} with task 'conversational'")
#         except Exception as e:
#             logger.exception("❌ Failed to initialize Hugging Face LLM.")
#             raise RuntimeError("Initialization failed") from e

#     def generate(self, prompt: str) -> str:
#         """
#         Generate a conversational-style response using the model.
#         """
#         if not prompt.strip():
#             logger.warning("⚠️ Empty prompt provided.")
#             return "⚠️ No prompt provided."

#         if self.enable_cache and prompt in self.cache:
#             logger.info("📦 Returning cached result.")
#             return self.cache[prompt]

#         try:
#             logger.debug(f"💬 Generating response from {self.model_name} (conversational) for prompt: {prompt[:100]}...")
#             response = self.llm.invoke(prompt)
#             output = str(response).strip().strip('"')

#             if self.enable_cache:
#                 self.cache[prompt] = output

#             logger.info("✅ Response generation successful.")
#             return output

#         except Exception as e:
#             logger.exception("❌ Error generating response.")
#             return f"❌ Error generating response from model '{self.model_name}' (conversational): {e}"







#This is the working code
# import os
# import logging
# from typing import Optional, List, Dict
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# # Load environment variables
# load_dotenv()

# # Configure logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# class HuggingFaceLLM:
#     """
#     Direct wrapper for Hugging Face's InferenceClient using chat_completion for conversational models.
#     """

#     def __init__(
#         self,
#         model_name: Optional[str] = None,
#         temperature: float = 0.3,
#         max_tokens: int = 1024,
#         top_p: float = 0.9,
#         enable_cache: bool = False
#     ):
#         self.model_name = model_name or os.getenv("LLM_MODEL")
#         self.api_key = os.getenv("HUGGINGFACE_API_KEY")

#         if not self.model_name:
#             raise ValueError("❌ Model name not provided. Set `LLM_MODEL` in .env or pass explicitly.")
#         if not self.api_key:
#             raise ValueError("❌ Hugging Face API key not found. Set `HUGGINGFACE_API_KEY` in .env")

#         self.temperature = temperature
#         self.max_tokens = max_tokens
#         self.top_p = top_p
#         self.enable_cache = enable_cache
#         self.cache = {}

#         try:
#             self.client = InferenceClient(model=self.model_name, token=self.api_key)
#             logger.info(f"✅ InferenceClient initialized for model: {self.model_name}")
#         except Exception as e:
#             logger.exception("❌ Failed to initialize InferenceClient.")
#             raise RuntimeError("Initialization failed") from e

#     def generate(self, user_prompt: str, system_prompt: str = "You are a helpful writing assistant.") -> str:
#         """
#         Generate a response from a conversational model using chat_completion.
#         """
#         if not user_prompt.strip():
#             logger.warning("⚠️ Empty prompt provided.")
#             return "⚠️ No prompt provided."

#         if self.enable_cache and user_prompt in self.cache:
#             logger.info("📦 Returning cached output.")
#             return self.cache[user_prompt]

#         try:
#             messages: List[Dict[str, str]] = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]

#             logger.debug(f"💬 Sending messages to model {self.model_name}: {messages}")
#             response = self.client.chat_completion(
#                 messages=messages,
#                 max_tokens=self.max_tokens,
#                 temperature=self.temperature,
#                 top_p=self.top_p
#             )

#             output = response.choices[0].message.content.strip()


#             if self.enable_cache:
#                 self.cache[user_prompt] = output

#             logger.info("✅ Chat response generated successfully.")
#             return output

#         except Exception as e:
#             logger.exception("❌ Error generating chat response.")
#             return f"❌ Error generating response from model '{self.model_name}': {e}"




#Ollama plus HuggingFace Inference support


import os
import logging
import json
import requests
from typing import Optional, List, Dict
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import streamlit as st

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class HuggingFaceLLM:
    """
    Wrapper for Hugging Face InferenceClient with fallback to local Ollama.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        enable_cache: bool = False,
        use_ollama_fallback: bool = True,
        ollama_model_name: str = "llama3"
    ):
        self.model_name = model_name or os.getenv("LLM_MODEL")
        self.api_key = os.getenv("HUGGINGFACE_API_KEY",st.secrets.get("HUGGINGFACE_API_KEY"))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.enable_cache = enable_cache
        self.use_ollama_fallback = use_ollama_fallback
        self.ollama_model_name = ollama_model_name
        self.cache = {}

        if not self.model_name:
            raise ValueError("❌ Model name not provided. Set `LLM_MODEL` in .env or pass explicitly.")
        if not self.api_key:
            raise ValueError("❌ Hugging Face API key not found. Set `HUGGINGFACE_API_KEY` in .env")

        try:
            self.client = InferenceClient(model=self.model_name, token=self.api_key)
            logger.info(f"✅ InferenceClient initialized for model: {self.model_name}")
        except Exception as e:
            logger.exception("❌ Failed to initialize InferenceClient.")
            raise RuntimeError("Initialization failed") from e

    def _run_ollama(self, prompt: str) -> str:
        """
        Fallback to local Ollama if Hugging Face inference fails.
        """
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.ollama_model_name, "prompt": prompt}
            )
            raw_text = response.text.strip()
            print("🔎 Ollama raw response:", raw_text)

            # Try to parse valid JSON if present on first line
            try:
                parsed = json.loads(raw_text.splitlines()[0])
                output = parsed.get("response", "").strip()
                return output or "⚠️ Ollama returned empty JSON response."
            except Exception:
                logger.warning("⚠️ Ollama response not valid JSON, using raw text fallback.")
                return raw_text

        except Exception as e:
            logger.error(f"❌ Ollama fallback failed: {e}")
            return f"❌ Unable to generate response from Ollama: {e}"

    def generate(self, user_prompt: str, system_prompt: str = "You are a helpful writing assistant.") -> str:
        """
        Attempt Hugging Face chat completion; fallback to Ollama if enabled and necessary.
        """
        if not user_prompt.strip():
            logger.warning("⚠️ Empty prompt provided.")
            return "⚠️ No prompt provided."

        if self.enable_cache and user_prompt in self.cache:
            logger.info("📦 Returning cached output.")
            return self.cache[user_prompt]

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logger.debug(f"💬 Sending messages to model {self.model_name}: {messages}")
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            output = response.choices[0].message.content.strip()

        except Exception as e:
            logger.exception("❌ Hugging Face chat completion failed.")
            if self.use_ollama_fallback:
                logger.warning("🔁 Switching to Ollama fallback...")
                output = self._run_ollama(user_prompt)
            else:
                output = f"❌ Error generating response from model '{self.model_name}': {e}"

        if self.enable_cache:
            self.cache[user_prompt] = output

        logger.info("✅ Response generated successfully.")
        return output
