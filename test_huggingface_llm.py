import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()  # ✅ Load .env values

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.environ["HF_TOKEN"]
)

response = client.chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of India?"}
    ],
)

print("✅ Response:", response.choices[0].message["content"])
