# src/vector_store.py
import faiss
import numpy as np
from typing import List, Dict, Optional, Callable
import pickle
import os


class VectorStore:
    def __init__(self, dimension: int = 768, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.document_store: Dict[int, Dict] = {}

    def _create_index(self):
        if self.index_type == "flat":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            index.nprobe = 10
            return index
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add_documents(self, document_chunks: List[Dict]):
        """Add document chunks to the vector store."""
        if not document_chunks:
            return

        embeddings = np.array(
            [chunk["embedding"] for chunk in document_chunks], dtype=np.float32
        )
        # embeddings are already normalized in DocumentProcessor, but normalize again defensively
        faiss.normalize_L2(embeddings)

        start_idx = len(self.document_store)
        self.index.add(embeddings)

        for i, chunk in enumerate(document_chunks):
            idx = start_idx + i
            self.document_store[idx] = chunk

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Optional[Callable[[Dict], bool]] = None,
    ) -> List[Dict]:
        """Search for similar documents with optional metadata filtering."""
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        results: List[Dict] = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            if idx not in self.document_store:
                continue
            doc = self.document_store[idx].copy()
            doc["score"] = float(scores[0][i])

            if metadata_filter is not None and not metadata_filter(doc):
                continue

            results.append(doc)

        return results

    def save(self, path: str):
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        faiss.write_index(self.index, f"{path}.index")

        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "document_store": self.document_store,
        }
        with open(f"{path}.docs", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """Load a vector store from disk."""
        with open(f"{path}.docs", "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            dimension=meta["dimension"],
            index_type=meta.get("index_type", "flat"),
        )
        instance.index = faiss.read_index(f"{path}.index")
        instance.document_store = meta["document_store"]
        return instance
