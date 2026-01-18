# ingest/chunk_text.py
from __future__ import annotations

from typing import Dict, Iterable, List


def _tokenize(text: str) -> List[str]:
    return text.split()


def chunk_text_records(
    records: Iterable[Dict[str, str]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, str]]:
    """Chunk records into overlapping windows.

    Uses a simple whitespace tokenizer so chunks are deterministic and fast.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    chunks: List[Dict[str, str]] = []
    for record in records:
        tokens = _tokenize(record.get("text", ""))
        if not tokens:
            continue

        step = chunk_size - overlap
        chunk_id = 0
        for start in range(0, len(tokens), step):
            window = tokens[start : start + chunk_size]
            if not window:
                continue
            chunks.append(
                {
                    "doc_id": record["doc_id"],
                    "chunk_id": f"{record['doc_id']}_c{chunk_id}",
                    "text": " ".join(window),
                }
            )
            chunk_id += 1
            if start + chunk_size >= len(tokens):
                break
    return chunks