# retrieval/hybrid_ranker.py
from __future__ import annotations

from typing import Dict, Iterable, List


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_rank(
    dense_results: Iterable[Dict[str, object]],
    sparse_results: Iterable[Dict[str, object]],
    alpha: float = 0.6,
    top_k: int = 5,
) -> List[Dict[str, object]]:
    """Blend dense and sparse results using normalized scores."""
    dense_list = list(dense_results)
    sparse_list = list(sparse_results)

    dense_scores = _normalize([float(item["score"]) for item in dense_list])
    sparse_scores = _normalize([float(item["score"]) for item in sparse_list])

    dense_map = {}
    for item, norm_score in zip(dense_list, dense_scores):
        dense_map[item["chunk_id"]] = {
            **item,
            "dense_score": norm_score,
        }

    sparse_map = {}
    for item, norm_score in zip(sparse_list, sparse_scores):
        sparse_map[item["chunk_id"]] = {
            **item,
            "sparse_score": norm_score,
        }

    combined: Dict[str, Dict[str, object]] = {}
    for chunk_id in set(dense_map) | set(sparse_map):
        dense_item = dense_map.get(chunk_id)
        sparse_item = sparse_map.get(chunk_id)
        base = dense_item or sparse_item
        if base is None:
            continue
        dense_score = dense_item["dense_score"] if dense_item else 0.0
        sparse_score = sparse_item["sparse_score"] if sparse_item else 0.0
        combined[chunk_id] = {
            "doc_id": base["doc_id"],
            "chunk_id": chunk_id,
            "text": base["text"],
            "dense_score": dense_score,
            "sparse_score": sparse_score,
            "score": alpha * dense_score + (1.0 - alpha) * sparse_score,
        }

    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]