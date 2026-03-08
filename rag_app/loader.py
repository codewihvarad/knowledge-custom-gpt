"""
loader.py — Document loading with multi-format support.
Supports PDF (PyMuPDF) and plain text files.
"""
import os
import hashlib
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from loguru import logger


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: str) -> Tuple[List[Document], str]:
    """
    Load a document and return (documents, doc_id).
    doc_id is a SHA-256 hash of the file content for deduplication.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Compute deterministic doc_id
    with open(file_path, "rb") as f:
        doc_id = hashlib.sha256(f.read()).hexdigest()[:16]

    if ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    documents = loader.load()
    logger.info(
        "Loaded {} pages/chunks from '{}' (doc_id={})", len(documents), path.name, doc_id
    )
    return documents, doc_id