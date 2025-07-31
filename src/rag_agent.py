import os
import logging
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
import faiss
from tqdm import tqdm

from src.nvidia_embeddings import NVIDIAEmbeddings as NemoRetrieverEmbeddings
from src.llm_hf import HuggingFaceLLM
from src.checklist_extractor import ChecklistRulesExtractor
from src.pdf_document_loader import PDFDocumentLoader
import streamlit as st


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RAGAgent:
    """RAG Agent for rewriting documents based on checklist rules"""

    def __init__(self):
        try:
            self.embedder = NemoRetrieverEmbeddings()
            if not self.embedder.test_connection():
                raise ConnectionError("🚫 NVIDIA Embedding API connection test failed.")
        except Exception as e:
            logger.error(f"[Embedding Initialization Failed] {e}")
            raise

        self.llm = HuggingFaceLLM()
        self.vector_db = None
        self.retriever = None
        self.doc_chunks: List[Document] = []

def load_checklist(self, checklist_path: Optional[str] = None) -> bool:
    try:
        path = checklist_path or os.getenv("CHECKLIST_PDF") or st.secrets.get("CHECKLIST_PDF")

        st.write("📄 Path from secrets:", path)
        st.write("📂 Current working dir:", os.getcwd())
        st.write("📁 Contents of directory:", os.listdir("Data/Checklist"))
        st.write("📄 File exists:", os.path.isfile(path))

        if not path:
            raise ValueError("Checklist path not specified.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checklist PDF not found at: {path}")

        extractor = ChecklistRulesExtractor(path)
        rule_docs = extractor.get_rules_as_documents()

        self.vector_db = FAISS.from_documents(rule_docs, self.embedder)
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        logger.info(f"✅ Checklist loaded with {len(rule_docs)} rules.")
        return True
    except Exception as e:
        logger.error(f"[Checklist Load Failed] {e}")
        return False


    def load_documents(self, docs_folder: Optional[str] = None) -> List[Document]:
        path = docs_folder or os.getenv("DOCS_FOLDER",st.secrets.get("DOCS_FOLDER"))
        if not path or not os.path.isdir(path):
            logger.error(f"Document folder not found: {path}")
            return []

        try:
            loader = PDFDocumentLoader(path)
            self.doc_chunks = loader.load_and_split()
            logger.info(f"✅ Loaded and chunked {len(self.doc_chunks)} document sections.")
            return self.doc_chunks
        except Exception as e:
            logger.error(f"[Document Loading Failed] {e}")
            return []

    def create_document_index(self, documents: List[Document]) -> bool:
        try:
            if not documents:
                raise ValueError("No documents provided for indexing.")

            # Run embedding test again to confirm validity
            sample_embedding = self.embedder.embed_query("sample test")
            if not sample_embedding:
                raise RuntimeError("Embedding failed during index creation.")

            dim = len(sample_embedding)
            faiss_index = faiss.IndexFlatIP(dim)

            self.vector_db = FAISS.from_documents(documents, self.embedder, faiss_index)
            self.retriever = self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 20}
            )
            logger.info("✅ Document index created successfully.")
            return True
        except Exception as e:
            logger.error(f"[Index Creation Failed] {e}")
            return False

    def generate_rewritten_content(self, document: Document) -> str:

        """
    Refines a narration block using internal checklist rules and LLM-based rewriting.
    Expects `document.page_content` to contain narration,
    and optionally `document.metadata["refine_hint"]` for user customization.
    """
        
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized.")

            # 📌 Retrieve checklist rules
            top_rules = self.retriever.get_relevant_documents(document.page_content[:1000])
            checklist_rules = "\n".join(f"- {doc.page_content}" for doc in top_rules)

            # 🧾 Original narration and hint (if any)
            original_narration = document.page_content.strip()
            refine_hint = document.metadata.get("refine_hint", "Make it concise and convert to bullet format.")
            
#             prompt = f"""
# You are refining the following narration using internal checklist rules.

# Checklist Rules:
# {checklist_rules}

# Original Narration:
# {original_narration}

# User Request:
# {refine_hint}

# Instructions:
# - Do NOT use markdown headings (no #, ##).
# - DO format each idea as a clear bullet point, starting with •
# - Do NOT include formatting symbols like *, _, ~, or backticks.
# - Keep language concise and instructional (tutorial-style).
# - Return only the refined narration in bullet format — no preamble or explanation.
# """

            
            prompt = f"""
                You are a professional editor rewriting technical documents based on rules.

                Checklist Rules:
                {checklist_rules}

                Original Content:
                {original_narration}

                User Request:
                {refine_hint}

                Rewrite the content using clear Markdown structure with headings, bullet points, and clarity:
                """

            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"[Rewrite Failed] {e}")
            return f"⚠️ Error generating content: {e}"
        

        


#             prompt = f"""<|system|>
# You are a professional editor rewriting documents based on these writing rules.

# ## Checklist Rules:
# {rules_md}
# </s>
# <|user|>
# Original Content:
# {document.page_content}

# Rewritten Content (markdown format):
# </s>
# <|assistant|>"""

    def process_documents(self, output_dir: Optional[str] = None) -> bool:
        out_path = output_dir or os.getenv("OUTPUT_DIR", "./output", st.secrets.get("OUTPUT_DIR", "./output"))
        os.makedirs(out_path, exist_ok=True)

        if not self.doc_chunks:
            logger.warning("⚠️ No document chunks to process.")
            return False

        success_count = 0
        for doc in tqdm(self.doc_chunks, desc="Rewriting PDFs"):
            try:
                rewritten = self.generate_rewritten_content(doc)

                base_name = str(doc.metadata.get("filename", "unnamed")).replace(".pdf", "")
                chunk_id = doc.metadata.get("chunk_id", 0)
                filename = f"{base_name}_chunk{chunk_id}.md"

                full_path = os.path.join(out_path, filename)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(rewritten)
                success_count += 1
            except Exception as e:
                logger.warning(f"[Chunk Save Failed] {e}")

        logger.info(f"✅ Successfully processed {success_count}/{len(self.doc_chunks)} chunks.")
        return success_count > 0
