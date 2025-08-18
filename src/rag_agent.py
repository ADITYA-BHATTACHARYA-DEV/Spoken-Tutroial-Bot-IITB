# import os
# import logging
# from typing import List, Optional

# from langchain.schema import Document
# from langchain_community.vectorstores import FAISS
# import faiss
# from tqdm import tqdm

# from src.nvidia_embeddings import NVIDIAEmbeddings as NemoRetrieverEmbeddings
# from src.llm_hf import HuggingFaceLLM
# from src.checklist_extractor import ChecklistRulesExtractor
# from src.pdf_document_loader import PDFDocumentLoader
# import streamlit as st


# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# class RAGAgent:
#     """RAG Agent for rewriting documents based on checklist rules"""

#     def __init__(self):
#         try:
#             self.embedder = NemoRetrieverEmbeddings()
#             if not self.embedder.test_connection():
#                 raise ConnectionError("🚫 NVIDIA Embedding API connection test failed.")
#         except Exception as e:
#             logger.error(f"[Embedding Initialization Failed] {e}")
#             raise

#         self.llm = HuggingFaceLLM()
#         self.vector_db = None
#         self.retriever = None
#         self.doc_chunks: List[Document] = []

#     def load_checklist(self, checklist_path: Optional[str] = None) -> bool:
#         try:
#             path = checklist_path or os.getenv("CHECKLIST_PDF") or st.secrets.get("CHECKLIST_PDF")

#             if not path or not os.path.exists(path):
#                 raise FileNotFoundError(f"Checklist PDF not found: {path}")

#             extractor = ChecklistRulesExtractor(path)
#             rule_docs = extractor.get_rules_as_documents()

#             self.vector_db = FAISS.from_documents(rule_docs, self.embedder)
#             self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

#             logger.info(f"✅ Checklist loaded with {len(rule_docs)} rules.")
#             return True
#         except Exception as e:
#             logger.error(f"[Checklist Load Failed] {e}")
#             return False

#     def load_documents(self, docs_folder: Optional[str] = None) -> List[Document]:
#         path = docs_folder or os.getenv("DOCS_FOLDER",st.secrets.get("DOCS_FOLDER"))
#         if not path or not os.path.isdir(path):
#             logger.error(f"Document folder not found: {path}")
#             return []

#         try:
#             loader = PDFDocumentLoader(path)
#             self.doc_chunks = loader.load_and_split()
#             logger.info(f"✅ Loaded and chunked {len(self.doc_chunks)} document sections.")
#             return self.doc_chunks
#         except Exception as e:
#             logger.error(f"[Document Loading Failed] {e}")
#             return []

#     def create_document_index(self, documents: List[Document]) -> bool:
#         try:
#             if not documents:
#                 raise ValueError("No documents provided for indexing.")

#             # Run embedding test again to confirm validity
#             sample_embedding = self.embedder.embed_query("sample test")
#             if not sample_embedding:
#                 raise RuntimeError("Embedding failed during index creation.")

#             dim = len(sample_embedding)
#             faiss_index = faiss.IndexFlatIP(dim)

#             self.vector_db = FAISS.from_documents(documents, self.embedder, faiss_index)
#             self.retriever = self.vector_db.as_retriever(
#                 search_type="mmr",
#                 search_kwargs={"k": 4, "fetch_k": 20}
#             )
#             logger.info("✅ Document index created successfully.")
#             return True
#         except Exception as e:
#             logger.error(f"[Index Creation Failed] {e}")
#             return False

#     def generate_rewritten_content(self, document: Document) -> str:

#         """
#     Refines a narration block using internal checklist rules and LLM-based rewriting.
#     Expects `document.page_content` to contain narration,
#     and optionally `document.metadata["refine_hint"]` for user customization.
#     """
        
#         try:
#             if not self.retriever:
#                 raise ValueError("Retriever not initialized.")

#             # 📌 Retrieve checklist rules
#             top_rules = self.retriever.get_relevant_documents(document.page_content[:1000])
#             checklist_rules = "\n".join(f"- {doc.page_content}" for doc in top_rules)

#             # 🧾 Original narration and hint (if any)
#             original_narration = document.page_content.strip()
#             refine_hint = document.metadata.get("refine_hint", "Make it concise and convert to bullet format.")
            
# #             prompt = f"""
# # You are refining the following narration using internal checklist rules.

# # Checklist Rules:
# # {checklist_rules}

# # Original Narration:
# # {original_narration}

# # User Request:
# # {refine_hint}

# # Instructions:
# # - Do NOT use markdown headings (no #, ##).
# # - DO format each idea as a clear bullet point, starting with •
# # - Do NOT include formatting symbols like *, _, ~, or backticks.
# # - Keep language concise and instructional (tutorial-style).
# # - Return only the refined narration in bullet format — no preamble or explanation.
# # """

            
#             prompt = f"""
#                 You are a professional editor rewriting technical documents based on rules.

#                 Checklist Rules:
#                 {checklist_rules}

#                 Original Content:
#                 {original_narration}

#                 User Request:
#                 {refine_hint}

#                 Rewrite the content using clear Markdown structure with headings, bullet points, and clarity:
#                 """

#             return self.llm.generate(prompt)
#         except Exception as e:
#             logger.error(f"[Rewrite Failed] {e}")
#             return f"⚠️ Error generating content: {e}"
        

        


# #             prompt = f"""<|system|>
# # You are a professional editor rewriting documents based on these writing rules.

# # ## Checklist Rules:
# # {rules_md}
# # </s>
# # <|user|>
# # Original Content:
# # {document.page_content}

# # Rewritten Content (markdown format):
# # </s>
# # <|assistant|>"""

#     def process_documents(self, output_dir: Optional[str] = None) -> bool:
#         out_path = output_dir or os.getenv("OUTPUT_DIR", "./output", st.secrets.get("OUTPUT_DIR", "./output"))
#         os.makedirs(out_path, exist_ok=True)

#         if not self.doc_chunks:
#             logger.warning("⚠️ No document chunks to process.")
#             return False

#         success_count = 0
#         for doc in tqdm(self.doc_chunks, desc="Rewriting PDFs"):
#             try:
#                 rewritten = self.generate_rewritten_content(doc)

#                 base_name = str(doc.metadata.get("filename", "unnamed")).replace(".pdf", "")
#                 chunk_id = doc.metadata.get("chunk_id", 0)
#                 filename = f"{base_name}_chunk{chunk_id}.md"

#                 full_path = os.path.join(out_path, filename)
#                 with open(full_path, "w", encoding="utf-8") as f:
#                     f.write(rewritten)
#                 success_count += 1
#             except Exception as e:
#                 logger.warning(f"[Chunk Save Failed] {e}")

#         logger.info(f"✅ Successfully processed {success_count}/{len(self.doc_chunks)} chunks.")
#         return success_count > 0



import os
import re
import logging
from typing import List, Optional, Dict

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
    """RAG Agent for rewriting documents based on checklist rules."""

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
        self.checklist: Optional[dict] = None

    def load_checklist(self, checklist_path: Optional[str] = None) -> bool:
        try:
            path = checklist_path or os.getenv("CHECKLIST_PDF") or st.secrets.get("CHECKLIST_PDF")
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Checklist PDF not found at: {path}")

            extractor = ChecklistRulesExtractor(path)
            self.checklist = extractor.extract_rules()

            rule_docs = extractor.get_rules_as_documents()
            self.vector_db = FAISS.from_documents(rule_docs, self.embedder)
            self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

            st.write("📄 Path resolved to:", path)
            st.write("📂 Working directory:", os.getcwd())
            st.write("📁 Checklist folder contents:", os.listdir(os.path.dirname(path)))

            logger.info(f"✅ Checklist loaded with {len(rule_docs)} rules.")
            return True
        except Exception as e:
            logger.error(f"[Checklist Load Failed] {e}")
            return False

    def load_documents(self, docs_folder: Optional[str] = None) -> List[Document]:
        path = docs_folder or os.getenv("DOCS_FOLDER") or st.secrets.get("DOCS_FOLDER")
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

            sample_embedding = self.embedder.embed_query("sample test")
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
        try:
            if not self.retriever:
                raise ValueError("Retriever not initialized.")

            top_rules = self.retriever.get_relevant_documents(document.page_content[:1000])
            checklist_rules = "\n".join(f"- {doc.page_content.strip()}" for doc in top_rules)

            original_narration = document.page_content.strip()
            refine_hint = document.metadata.get("refine_hint", "Convert to human-friendly narration with visual cues.")

            # ✂️ Segment narration into logical chunks
            segments = re.split(r"\n{2,}", original_narration)
            segmented_narration = "\n".join(
                f"Slide {i+1}:\n{seg.strip()}" for i, seg in enumerate(segments) if seg.strip()
            )

            prompt = f"""
You are a professional editor rewriting technical tutorials for spoken narration.
Your task is to convert the original content into a slide-by-slide format suitable
for voice-over delivery.

Use the following checklist rules to guide your rewrite:
{checklist_rules}

Original Content (Segmented by Slide):
{segmented_narration}

User Request:
{refine_hint}

🔧 Rewrite Instructions:

- Format the output as a Markdown table with two columns: **Visual Cue** and **Narration**
- Each row must represent a distinct slide or visual moment in the tutorial

📌 Visual Cue Guidelines:
• Do not use bullet points in this column
• Describe the key visual, interaction, or code snippet in one or two concise sentences
• If code is present, include it as-is or summarize it clearly
• For long code blocks, summarize structure and use sub-bullets in the Narration column only

🗣️ Narration Guidelines:
• Use formal, instructional language suitable for academic tutorials
• Maintain a tutorial-style tone appropriate for academic learners
• Avoid casual expressions or conversational tone
• Use phrases such as:
  • “Let us now”
  • “Observe that”
  • “To do this”
  • “You should”
• Narration may exceed 80 characters
• If a sentence exceeds 80 characters, break it into sub-bullets
• Each bullet or sub-bullet must:
  • Begin with “•”
  • Appear on a new line
  • Be placed only in the Narration column
• Do not use inline separators such as <br>, commas, or semicolons for bullet formatting
• Maintain technical accuracy and completeness
• Avoid unexplained jargon
• Ensure clarity and consistency throughout

📄 Formatting Enforcement:
• Ensure the Markdown table is syntactically valid
• Do not merge Visual Cue and Narration content into one column
• Do not use multiline cells in the Visual Cue column
• Bullet points must be rendered as separate lines in the Narration column only

🎯 Output Format Example:

Visual Cue | Narration
-----------|----------
Title Slide | Welcome to this tutorial on camera view settings in Blender.
Learning Objectives | In this tutorial, we will learn to:
                   • Move the camera
                   • Use fly mode
                   • Adjust views
Cursor on the interface | Blender is open. Let us locate the camera controls.
Press numpad 0 | To switch to camera view:
               • Press numpad 0
               • Or go to View > Camera > Active Camera
Code: hist(movies$runtime, ...) | This command creates a histogram of runtimes.
Highlight hist arguments | We have used the following arguments:
                         • main for title
                         • xlab for x-axis label
                         • xlim to set range
                         • col to set color
Code: <dl> with multiple <dt> and <dd> tags | This code defines a description list.
Sub-bullets: | • Dry fruits: Cashew, Almond, Apricot, Dates, Fig
             • Fresh fruits: Orange, Mango, Apple, Grapes, Melon, Banana, Cherry,
               Blueberry, Apricot, Pineapple, Dragon Fruit, Avocados, Kiwi, Pear,
               Pomegranate, Plum, Papaya
Highlight <dt> and <dd> usage | <dt> is used for category titles.
                              • <dd> lists the items under each category.
Instructional Note | To practise this, you should know basic HTML and CSS.
                   • If not, review the tutorials linked in the resource section.

Now, rewrite the content accordingly:
"""






            rewritten_output = self.llm.generate(prompt)

            if not rewritten_output.strip().startswith("Visual Cue | Narration"):
                logger.warning("[Output Format Warning] Response may not be a valid Markdown table.")

            return rewritten_output
        except Exception as e:
            logger.error(f"[Rewrite Failed] {e}")
            return f"⚠️ Error generating content: {e}"

    def parse_markdown_table(self, md_text: str) -> List[Dict[str, str]]:
        rows = []
        lines = md_text.strip().split("\n")
        for line in lines[2:]:  # Skip header and separator
            if "|" in line:
                parts = [cell.strip() for cell in line.split("|")]
                if len(parts) >= 2:
                    rows.append({
                        "visual_cue": parts[0],
                        "narration": parts[1]
                    })
        return rows
    

    def parse_markdown_table(self, md_text: str) -> List[Dict[str, str]]:
        rows = []
        lines = md_text.strip().split("\n")
        for line in lines[2:]:  # Skip header and separator
            if "|" in line:
                parts = [cell.strip() for cell in line.split("|")]
                if len(parts) >= 2:
                    rows.append({
                    "Visual Cue": parts[0],
                    "Narration": parts[1]
                })
        return rows


    def process_documents(self, output_dir: Optional[str] = None) -> bool:
        out_path = output_dir or os.getenv("OUTPUT_DIR") or st.secrets.get("OUTPUT_DIR") or "./output"
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
