# ingest/parse_docs.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def load_pdf_pages(data_dir: str | Path) -> List[Dict[str, str]]:
    """Load PDFs and return one record per page.

    Each record has a stable doc_id tied to the file and page number.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    records: List[Dict[str, str]] = []
    for pdf_path in sorted(data_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        stem = pdf_path.stem
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue
            records.append(
                {
                    "doc_id": f"{stem}_p{page_num}",
                    "text": text,
                }
            )
    return records