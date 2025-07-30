import os
from dotenv import load_dotenv
load_dotenv()

from src.rag_agent import RAGAgent
import sys
sys.path.append("src")

def test_rag_pipeline():
    # Set environment variables or pass directly
    os.environ["CHECKLIST_PDF"] = "./data/checklist/chkList.pdf"
    os.environ["DOCS_FOLDER"] = "./data/docs/"
    os.environ["OUTPUT_DIR"] = "./output_test"

    agent = RAGAgent()

    # Step 1: Load checklist
    assert agent.load_checklist(), "❌ Failed to load checklist rules"

    # Step 2: Load documents
    chunks = agent.load_documents()
    assert len(chunks) > 0, "❌ No document chunks loaded"

    # Step 3: Create document index
    assert agent.create_document_index(chunks), "❌ Failed to create document index"

    # Step 4: Process and rewrite documents
    success = agent.process_documents()
    assert success, "❌ No documents were successfully processed"

    print("✅ RAGAgent test passed")

if __name__ == "__main__":
    test_rag_pipeline()
