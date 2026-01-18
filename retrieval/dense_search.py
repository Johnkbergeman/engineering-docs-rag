# retrieval/dense_search.py
from __future__ import annotations

from typing import Dict, Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def _l2_to_similarity(distances: np.ndarray) -> np.ndarray:
    # Smaller L2 means closer; convert to bounded similarity for easier mixing.
    return 1.0 / (1.0 + distances)


class DenseIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: List[Dict[str, str]] = []

    def embed_texts(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype="float32")

        embeddings: List[np.ndarray] = []
        total = len(text_list)
        for start in tqdm(
            range(0, total, batch_size),
            desc="Embedding chunks",
            unit="batch",
        ):
            end = min(start + batch_size, total)
            batch = text_list[start:end]
            batch_vecs = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(np.asarray(batch_vecs, dtype="float32"))

        return np.vstack(embeddings)

    def build_index(self, chunks: Iterable[Dict[str, str]]) -> None:
        self.metadata = []
        texts: List[str] = []
        for chunk in chunks:
            self.metadata.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )
            texts.append(chunk["text"])

        if not texts:
            self.index = None
            return

        vectors = self.embed_texts(texts)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        if not self.index or not self.metadata:
            return []

        query_vec = self.embed_texts([query])
        distances, indices = self.index.search(query_vec, top_k)
        scores = _l2_to_similarity(distances[0])

        results: List[Dict[str, object]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores)):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                }
            )
        return results
