import os
from dotenv import load_dotenv
from src.nvidia_embeddings import NVIDIAEmbeddings

def test_nvidia_embeddings():
    load_dotenv()

    try:
        embedder = NVIDIAEmbeddings()
        test_text = "This is a test embedding."
        vector = embedder.embed_query(test_text)

        if vector and isinstance(vector, list) and all(isinstance(x, float) for x in vector):
            print("✅ NVIDIA Embedding API Test Passed. Vector length:", len(vector))
        else:
            print("❌ Unexpected embedding format or empty vector.")

    except Exception as e:
        print(f"❌ NVIDIA Embedding API Test Failed: {e}")

if __name__ == "__main__":
    test_nvidia_embeddings()
