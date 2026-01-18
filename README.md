# Engineering Docs RAG (Hybrid Retrieval)

This project is a small, production-style retrieval system for technical engineering documents. It combines semantic search (dense embeddings with FAISS) and keyword search (BM25) to surface relevant sections from PDFs that are difficult to search with keywords alone.

The goal is to demonstrate how hybrid retrieval works in real engineering documentation, where terminology, symbols, and phrasing do not always map cleanly to pure semantic search.

## What this does

At a high level, the system:

- Ingests PDF engineering documents
- Breaks text into overlapping chunks
- Indexes those chunks using both dense and sparse retrieval
- Blends results from both approaches into a single ranked list

The code is intentionally kept simple and local-only, with no external services or orchestration frameworks, so the full pipeline is easy to inspect and reason about.

## How it is structured

The retrieval pipeline follows these steps:

1. **Document ingestion** (`ingest/parse_docs.py`)  
   Reads PDFs from `data/sample_docs/` and extracts text page by page.

2. **Chunking** (`ingest/chunk_text.py`)  
   Splits text into overlapping chunks (default: 500 tokens with 100-token overlap) to balance context size and retrieval precision.

3. **Dense retrieval** (`retrieval/dense_search.py`)  
   Embeds chunks using the `all-MiniLM-L6-v2` sentence-transformer and indexes them with FAISS for semantic similarity search.

4. **Sparse retrieval** (`retrieval/bm25.py`)  
   Builds a BM25 index over the same chunks to capture exact keyword matches and technical terminology.

5. **Hybrid ranking** (`retrieval/hybrid_ranker.py`)  
   Normalizes and combines dense and sparse scores using a configurable weighting factor.

6. **Query engine** (`app/query_engine.py`)  
   Orchestrates indexing and retrieval, then runs a simple example query end-to-end.

## Repository layout

```
app/ query orchestration and example run
ingest/ PDF parsing and chunking
retrieval/ dense, sparse, and hybrid retrieval logic
evaluation/ simple retrieval metrics
data/sample_docs/ place PDF documents here

```