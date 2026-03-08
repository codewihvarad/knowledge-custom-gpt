"""
chunk.py — Smart recursive text splitter with metadata preservation.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import get_settings

settings = get_settings()


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of LangChain Documents into smaller overlapping chunks.
    Preserves and enriches metadata from the original documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata.setdefault("page", chunk.metadata.get("page", 0))

    logger.info(
        "Split {} document(s) into {} chunks (size={}, overlap={})",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks