# src/pipeline.py
import os
from typing import Dict, Optional

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.response_generator import ResponseGenerator


class RAGPipeline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        vector_db_path: str = "./vector_db",
        api_key: Optional[str] = None,
        claude_model: str = "claude-haiku-4-5-20251001",
        index_type: str = "flat",
        top_k: int = 5,
        fetch_k: int = 20,
        min_score: Optional[float] = None,
    ):
        """
        Orchestrates document ingestion, retrieval, and response generation.
        """
        self.document_processor = DocumentProcessor(model_name=model_name)

        # Try to load existing vector store or create a new one
        try:
            self.vector_store = VectorStore.load(vector_db_path)
            print(f"Loaded existing vector store from {vector_db_path}")
        except Exception:
            print(f"Creating new vector store at {vector_db_path}")
            self.vector_store = VectorStore(
                dimension=self.document_processor.embedding_dim,
                index_type=index_type,
            )

        self.query_processor = QueryProcessor(self.document_processor, self.vector_store)
        self.response_generator = ResponseGenerator(api_key=api_key, model=claude_model)
        self.vector_db_path = vector_db_path

        self.top_k = top_k
        self.fetch_k = fetch_k
        self.min_score = min_score

    def ingest_document(self, document_path: str, metadata: Optional[Dict] = None):
        """Ingest a document into the RAG pipeline."""
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")

        filename = os.path.basename(document_path)
        if metadata is None:
            metadata = {}
        metadata.setdefault("filename", filename)
        metadata.setdefault("source_path", document_path)

        # Simple text ingestion; you can extend this to PDF, etc.
        with open(document_path, "r", encoding="utf-8") as f:
            document = f.read()

        print(f"Processing document: {filename}")
        chunks, _ = self.document_processor.process_document(document, metadata)
        print(f"Created {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)
        self.vector_store.save(self.vector_db_path)
        print("Document ingested and vector store saved")

    def process_query(self, query: str) -> Dict:
        """Process a query end-to-end and generate a response."""
        query_result = self.query_processor.process_query(
            query,
            top_k=self.top_k,
            fetch_k=self.fetch_k,
            min_score=self.min_score,
        )

        context = query_result["context"]
        if not context.strip():
            # no good matches; still call Claude but make it clear
            context = "There is no relevant context available for this question."

        response = self.response_generator.generate_response(query, context)

        result = {
            "query": query,
            "response": response["response"],
            "retrieved_chunks": [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "metadata": r.get("metadata", {}),
                }
                for r in query_result["results"]
            ],
        }
        return result
