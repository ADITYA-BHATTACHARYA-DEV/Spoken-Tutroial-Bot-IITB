"""
Vector Database for RAG Agent
Handles local vector storage using FAISS for document embeddings
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .nvidia_embeddings import NVIDIAEmbeddings  # Ensure this implements embed_documents() and embed_query()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """Local vector database using FAISS for efficient similarity search"""

    def __init__(
        self,
        embeddings: NVIDIAEmbeddings,
        db_path: str = "./vector_db",
        index_name: str = "faiss_index"
    ):
        self.embeddings = embeddings
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.vectorstore: Optional[FAISS] = None

        self.db_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.db_path / f"{index_name}.faiss"
        self.metadata_path = self.db_path / f"{index_name}_metadata.pkl"

        logger.info(f"Initialized vector database at: {self.db_path}")

    def create_index(self, documents: List[Document]) -> bool:
        if not documents:
            logger.warning("No documents provided for indexing.")
            return False

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            logger.info(f"Creating vector index from {len(documents)} documents...")

            self.vectorstore = FAISS.from_documents(documents, embedding=self.embeddings)

            logger.info("✅ Vector index created successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False

    def save_index(self) -> bool:
        if not self.vectorstore:
            logger.error("No vector index to save.")
            return False

        try:
            self.vectorstore.save_local(str(self.db_path), self.index_name)
            logger.info("✅ Vector index saved successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            return False

    def load_index(self) -> bool:
        try:
            if not self.index_path.exists():
                logger.warning("No saved index found.")
                return False

            logger.info("Loading vector index from disk...")
            self.vectorstore = FAISS.load_local(
                str(self.db_path),
                embeddings=self.embeddings,
                index_name=self.index_name
            )
            logger.info("✅ Vector index loaded successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        if not documents:
            logger.warning("No documents provided to add.")
            return False

        try:
            if not self.vectorstore:
                logger.info("Index not loaded, creating new one.")
                return self.create_index(documents)

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            logger.info("✅ Documents added to vectorstore.")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        if not self.vectorstore:
            logger.error("No vector index loaded.")
            return []

        try:
            if score_threshold is not None:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                filtered = [doc for doc, score in results if score >= score_threshold]
                return filtered
            else:
                return self.vectorstore.similarity_search(query, k=k)

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not self.vectorstore:
            logger.error("No vector index loaded.")
            return []

        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Search with scores failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "No index loaded", "document_count": 0}

        try:
            return {
                "status": "Index loaded",
                "document_count": self.vectorstore.index.ntotal,
                "index_path": str(self.index_path),
                "index_exists": self.index_path.exists()
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "Error", "error": str(e)}

    def delete_index(self) -> bool:
        try:
            if self.index_path.exists():
                self.index_path.unlink()
                logger.info("Deleted FAISS index file.")
            if self.metadata_path.exists():
                self.metadata_path.unlink()
                logger.info("Deleted metadata file.")
            self.vectorstore = None
            return True
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False


def main():
    """Test the vector database"""
    from dotenv import load_dotenv
    load_dotenv()

    embeddings = NVIDIAEmbeddings()
    vector_db = VectorDatabase(embeddings)

    # Sample test documents
    test_docs = [
        Document(
            page_content="Artificial intelligence is the simulation of human intelligence by machines.",
            metadata={"source": "doc1.txt", "page": 1}
        ),
        Document(
            page_content="Machine learning enables systems to learn from data and improve from experience.",
            metadata={"source": "doc2.txt", "page": 2}
        )
    ]

    # Create index and save
    if vector_db.create_index(test_docs):
        vector_db.save_index()

        # Test similarity search
        query = "Tell me about artificial intelligence"
        results = vector_db.similarity_search(query, k=2)

        print(f"Search Results for: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content} | Metadata: {doc.metadata}")

        # Show stats
        print("Vector DB Stats:", vector_db.get_stats())


if __name__ == "__main__":
    main()
