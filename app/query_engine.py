# app/query_engine.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ingest.chunk_text import chunk_text_records
from ingest.parse_docs import load_pdf_pages
from retrieval.bm25 import BM25Searcher
from retrieval.dense_search import DenseIndexer
from retrieval.hybrid_ranker import hybrid_rank


def build_indexes(data_dir: Path) -> Dict[str, object]:
    records = load_pdf_pages(data_dir)
    chunks = chunk_text_records(records)

    dense = DenseIndexer()
    dense.build_index(chunks)

    sparse = BM25Searcher()
    sparse.build_index(chunks)

    return {
        "records": records,
        "chunks": chunks,
        "dense": dense,
        "sparse": sparse,
    }


def run_query(
    query: str,
    dense: DenseIndexer,
    sparse: BM25Searcher,
    top_k: int = 5,
    alpha: float = 0.6,
) -> List[Dict[str, object]]:
    dense_results = dense.search(query, top_k=top_k)
    sparse_results = sparse.search(query, top_k=top_k)
    return hybrid_rank(dense_results, sparse_results, alpha=alpha, top_k=top_k)


def main() -> None:
    data_dir = ROOT / "data" / "sample_docs"
    indexes = build_indexes(data_dir)

    if not indexes["chunks"]:
        print("No PDF pages found in data/sample_docs. Add PDFs and retry.")
        return

    query = "safety protocol for pressure valve installation"
    results = run_query(query, indexes["dense"], indexes["sparse"], top_k=5, alpha=0.6)

    print(f"Query: {query}")
    for rank, result in enumerate(results, start=1):
        preview = result["text"][:200].replace("\n", " ")
        print(
            f"{rank}. score={result['score']:.3f} doc={result['doc_id']} chunk={result['chunk_id']}\n"
            f"   {preview}"
        )


if __name__ == "__main__":
    main()