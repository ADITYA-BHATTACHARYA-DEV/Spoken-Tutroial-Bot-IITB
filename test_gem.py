from dotenv import load_dotenv
load_dotenv()

from src.gemini_llm import GeminiLLM

def test_gemini_llm():
    llm = GeminiLLM(model_name="gemini-1.5-pro")
    prompt = "Summarize the importance of AI in healthcare"
    result = llm.generate(prompt)
    print(result)

test_gemini_llm()
