# src/query_processor.py
from typing import List, Dict, Optional
import numpy as np


class QueryProcessor:
    def __init__(self, document_processor, vector_store):
        self.document_processor = document_processor
        self.vector_store = vector_store

    def _mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict],
        top_k: int,
        lambda_param: float = 0.7,
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance to balance relevance and diversity.
        Assumes each candidate has an 'embedding' field.
        """
        if not candidates:
            return []

        cand_embeddings = np.stack([c["embedding"] for c in candidates])
        # compute cosine similarity query–doc (they're already normalized)
        query_sim = cand_embeddings @ query_embedding.reshape(-1, 1)
        query_sim = query_sim.squeeze(-1)

        selected_indices: List[int] = []
        remaining_indices = list(range(len(candidates)))

        while remaining_indices and len(selected_indices) < top_k:
            mmr_scores = []
            for idx in remaining_indices:
                # diversity term: max similarity to any already selected doc
                if selected_indices:
                    selected_embs = cand_embeddings[selected_indices]
                    sim_to_selected = selected_embs @ cand_embeddings[idx].reshape(-1, 1)
                    sim_to_selected = sim_to_selected.max()
                else:
                    sim_to_selected = 0.0

                score = lambda_param * query_sim[idx] - (1 - lambda_param) * sim_to_selected
                mmr_scores.append((idx, score))

            # pick best candidate
            next_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        return [candidates[i] for i in selected_indices]

    def process_query(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        min_results: int = 3,
        min_score: Optional[float] = None,
    ) -> Dict:
        """
        Process a query and retrieve high-quality context.

        fetch_k: how many candidates to retrieve from FAISS before re-ranking.
        top_k: final number of chunks after MMR.
        min_score: optional absolute score cut-off; if None, a dynamic cut-off is used.
        """
        # 1. Query embedding
        query_embedding = self.document_processor.generate_embeddings([query])[0]

        # 2. Initial dense retrieval (high recall)
        raw_results = self.vector_store.search(
            query_embedding, top_k=fetch_k
        )

        if not raw_results:
            return {
                "query": query,
                "results": [],
                "context": "",
            }

        # 3. Dynamic score thresholding
        scores = np.array([r["score"] for r in raw_results])
        if min_score is None:
            # keep chunks above mean - 0.5 * std (tunable)
            mean = float(scores.mean())
            std = float(scores.std() if scores.std() > 0 else 0.0)
            dynamic_threshold = mean - 0.5 * std
        else:
            dynamic_threshold = min_score

        filtered = [r for r in raw_results if r["score"] >= dynamic_threshold]

        # ensure at least min_results
        if len(filtered) < min_results:
            filtered = raw_results[: max(min_results, top_k)]

        # 4. MMR for diversity
        reranked = self._mmr(query_embedding, filtered, top_k=top_k)

        # 5. Build context string
        context_parts: List[str] = []
        for i, r in enumerate(reranked):
            meta = r.get("metadata", {})
            source = meta.get("filename", meta.get("source_path", "unknown_source"))
            context_parts.append(
                f"[Chunk {i+1} | Source: {source}]\n{r['text']}\n"
            )
        context = "\n\n".join(context_parts)

        return {
            "query": query,
            "results": reranked,
            "context": context,
        }
