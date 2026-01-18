# evaluation/retrieval_metrics.py
from __future__ import annotations

from typing import Iterable, List, Set


def recall_at_k(retrieved_ids: Iterable[str], relevant_ids: Iterable[str], k: int) -> float:
    """Compute recall@k for a single query."""
    if k <= 0:
        return 0.0
    retrieved_list = list(retrieved_ids)[:k]
    relevant_set: Set[str] = set(relevant_ids)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in retrieved_list if item in relevant_set)
    return hits / len(relevant_set)


def synthetic_recall_demo() -> float:
    """Small demo to validate the metric behaves as expected."""
    relevant = {"doc_a", "doc_b"}
    retrieved = ["doc_c", "doc_b", "doc_d", "doc_a"]
    return recall_at_k(retrieved, relevant, k=3)


if __name__ == "__main__":
    print(f"Synthetic recall@3: {synthetic_recall_demo():.2f}")