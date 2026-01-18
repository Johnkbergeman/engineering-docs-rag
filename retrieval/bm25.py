# retrieval/bm25.py
from __future__ import annotations

import re
from typing import Dict, Iterable, List

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class BM25Searcher:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.metadata: List[Dict[str, str]] = []
        self.corpus_tokens: List[List[str]] = []

    def build_index(self, chunks: Iterable[Dict[str, str]]) -> None:
        self.metadata = []
        self.corpus_tokens = []

        for chunk in chunks:
            tokens = _tokenize(chunk["text"])
            if not tokens:
                continue
            self.metadata.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                }
            )
            self.corpus_tokens.append(tokens)

        if not self.corpus_tokens:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        if not self.bm25 or not self.metadata:
            return []

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        if scores.size == 0:
            return []

        top_indices = scores.argsort()[::-1][:top_k]
        results: List[Dict[str, object]] = []
        for rank, idx in enumerate(top_indices):
            meta = self.metadata[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "doc_id": meta["doc_id"],
                    "chunk_id": meta["chunk_id"],
                    "text": meta["text"],
                }
            )
        return results