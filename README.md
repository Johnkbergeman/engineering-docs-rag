# Industrial RAG System (Engineering Docs)

Hybrid retrieval system for engineering documents using dense embeddings (FAISS) and sparse keyword search (BM25).

## System overview
This project ingests PDF engineering documents, chunks them into overlapping text windows, and builds two retrieval indexes:
- Dense embeddings with sentence-transformers + FAISS
- Sparse keyword search with BM25

Queries are executed against both indexes, then blended into a single ranked list.

## Architecture
1. **Ingestion** (`ingest/parse_docs.py`): Read PDFs from `data/sample_docs/` and extract text page-by-page.
2. **Chunking** (`ingest/chunk_text.py`): Split each page into overlapping chunks (default 500 tokens, 100 overlap).
3. **Dense retrieval** (`retrieval/dense_search.py`): Embed chunks with `all-MiniLM-L6-v2`, index with FAISS.
4. **Sparse retrieval** (`retrieval/bm25.py`): Tokenize and rank with BM25.
5. **Hybrid ranking** (`retrieval/hybrid_ranker.py`): Normalize scores and combine them with configurable alpha.
6. **Query engine** (`app/query_engine.py`): Orchestrates indexing and retrieval.

## How to run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place PDF files in `data/sample_docs/`.
3. Run the query engine:
   ```bash
   python app/query_engine.py
   ```

## Status
In progress.