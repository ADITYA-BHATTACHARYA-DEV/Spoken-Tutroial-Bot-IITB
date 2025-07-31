import os
import time
import logging
import requests
from typing import List, Optional
from langchain.embeddings.base import Embeddings
import streamlit as st

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class NVIDIAEmbeddings(Embeddings):
    """
    NVIDIA LLaMA NemoRetriever Embeddings for LangChain.
    Connects to NVIDIA's API using your desired model.
    """

    def __init__(
        self,
        model_name: str = "nvidia/nv-embed-v1",  # 🧠 Your target model
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        batch_size: int = 32,
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY",st.secrets.get("NVIDIA_API_KEY"))
        if not self.api_key:
            raise ValueError("❌ NVIDIA_API_KEY is required (via env var or init arg)")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"✅ Initialized NVIDIAEmbeddings with model: {self.model_name}")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float"
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"📤 Attempt {attempt}: POST {url}")
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                logger.debug(f"🔎 Status: {response.status_code} | Response: {response.text}")

                if response.status_code == 200:
                    response_json = response.json()
                    return [item["embedding"] for item in response_json["data"]]

                elif response.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    logger.warning(f"⏱️ Rate limit hit. Sleeping for {wait}s")
                    time.sleep(wait)
                    continue

                else:
                    response.raise_for_status()

            except requests.RequestException as e:
                logger.error(f"❌ Request exception on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise RuntimeError("❌ Embedding failed after max retries.") from e
                time.sleep(1)

        raise RuntimeError("❌ Failed to get embeddings: max retries exceeded.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("⚠️ embed_documents called with empty input")
            return []

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.debug(f"📚 Embedding batch {i // self.batch_size + 1}")
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text:
            logger.warning("⚠️ embed_query called with empty text")
            return []
        return self._embed_batch([text])[0]

    def test_connection(self) -> bool:
        logger.info("🔌 Testing NVIDIA API connection...")
        try:
            result = self.embed_query("connection test")
            logger.info(f"✅ Connection successful. Embedding length: {len(result)}")
            return isinstance(result, list) and len(result) > 0
        except Exception as e:
            logger.exception("❌ Connection test failed")
            return False

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    embeddings = NVIDIAEmbeddings()
    connected = embeddings.test_connection()
    print("✅ Connected!" if connected else "❌ Failed to connect.")