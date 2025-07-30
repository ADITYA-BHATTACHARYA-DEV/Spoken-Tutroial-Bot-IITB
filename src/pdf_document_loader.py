"""
PDF Document Loader for RAG Agent
Handles loading and processing PDF documents from local folder
"""

import os
import logging
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDFDocumentLoader:
    """Handles loading and chunking of PDF documents"""

    def __init__(self, docs_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create folder if it doesn't exist
        self.docs_folder.mkdir(parents=True, exist_ok=True)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        logger.info(f"PDF loader initialized for: {self.docs_folder}")

    def load_documents(self) -> List[Document]:
        """Loads PDF files as LangChain Document objects"""
        if not self.docs_folder.exists():
            logger.error(f"Documents folder does not exist: {self.docs_folder}")
            return []

        pdf_files = list(self.docs_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_folder}")
            return []

        documents = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                pdf_docs = loader.load()

                for doc in pdf_docs:
                    doc.metadata.update({
                        "filename": pdf_file.name,
                        "filepath": str(pdf_file)
                    })

                documents.extend(pdf_docs)
                logger.info(f"Loaded {len(pdf_docs)} pages from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {pdf_file.name}: {str(e)}")

        logger.info(f"Loaded total {len(documents)} pages from {len(pdf_files)} files")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks"""
        if not documents:
            logger.warning("No documents to split")
            return []

        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(documents)

        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                "chunk_id": i,
                "chunk_size": len(doc.page_content)
            })

        logger.info(f"Generated {len(split_docs)} total chunks")
        return split_docs

    def load_and_split(self) -> List[Document]:
        """Convenience method to load and split documents"""
        docs = self.load_documents()
        return self.split_documents(docs)

    def get_stats(self, docs: List[Document]) -> dict:
        """Generate basic stats about documents"""
        total_chars = sum(len(doc.page_content) for doc in docs)
        source_files = set(doc.metadata.get("filename", "unknown") for doc in docs)

        return {
            "total_chunks": len(docs),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(docs) if docs else 0,
            "num_source_files": len(source_files),
            "source_files": list(source_files),
        }


# Optional test block
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    loader = PDFDocumentLoader(os.getenv("DOCS_FOLDER", "Data/Docs"))
    chunks = loader.load_and_split()
    stats = loader.get_stats(chunks)

    print("📊 Document Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
