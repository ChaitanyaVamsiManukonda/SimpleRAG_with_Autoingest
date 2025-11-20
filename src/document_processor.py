# src/document_processor.py
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple, Optional
import re
from nltk.tokenize import sent_tokenize
import hashlib


class DocumentProcessor:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_chars: int = 200,
        embedding_batch_size: int = 8,
    ):
        """
        chunk_size: approximate max tokens per chunk.
        chunk_overlap: approximate tokens of overlap between chunks.
        min_chunk_chars: avoid tiny chunks when possible.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.embedding_batch_size = embedding_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Infer embedding dimension from model
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            out = self.model(**dummy)
            self.embedding_dim = out.last_hidden_state.size(-1)

    def preprocess_text(self, text: str) -> str:
        # Basic normalization
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[“”]", '"', text)
        text = re.sub(r"[‘’]", "'", text)
        return text.strip()

    def create_document_chunks(
        self, document: str, metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Split document into overlapping, sentence-aware chunks."""
        document = self.preprocess_text(document)
        sentences = sent_tokenize(document)

        chunks: List[Dict] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_token_count = len(sentence_tokens)

            # If adding this sentence would exceed chunk_size, finalize current chunk
            if current_tokens + sentence_token_count > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                # avoid tiny chunks by greedily attaching one more sentence if needed
                if len(chunk_text) < self.min_chunk_chars and sentences:
                    # try to append next sentence if available
                    pass  # keep as-is; we already grew additively

                chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata or {},
                    }
                )

                # build overlap
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                while current_chunk and overlap_tokens < self.chunk_overlap:
                    s = current_chunk.pop()
                    tks = self.tokenizer.tokenize(s)
                    overlap_tokens += len(tks)
                    overlap_sentences.insert(0, s)

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_token_count

        # last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata or {},
                }
            )

        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts, L2-normalized."""
        embeddings: List[np.ndarray] = []
        batch_size = self.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.chunk_size,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                model_output = self.model(**inputs)

            attention_mask = inputs["attention_mask"]
            token_embeddings = model_output.last_hidden_state

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_emb = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.extend(batch_emb)

        emb_array = np.array(embeddings, dtype=np.float32)
        # normalize here so FAISS can safely use inner product
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-12
        emb_array = emb_array / norms
        return emb_array

    def process_document(
        self, document: str, metadata: Optional[Dict] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """Process a document into chunks and embeddings."""
        chunks = self.create_document_chunks(document, metadata)
        texts = [c["text"] for c in chunks]
        embeddings = self.generate_embeddings(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks, embeddings
