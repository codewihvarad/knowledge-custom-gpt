"""
chain.py — Full RAG pipeline: ingest + query.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

from chunk import chunk_documents
from config import get_settings
from database import Neo4jDatabase
from embedding import get_embedder
from loader import load_document
from ollama_model import build_rag_chain, format_context
from retriever import Neo4jRetriever

settings = get_settings()


# ── Ingestion ──────────────────────────────────────────────────────────────


def ingest_document(file_path: str) -> Dict[str, Any]:
    """
    Full ingestion pipeline:
      load → chunk → embed → store in Neo4j
    Returns a summary dict.
    """
    path = Path(file_path)
    logger.info("Starting ingestion for: {}", path.name)

    # 1. Load
    documents, doc_id = load_document(file_path)

    # 2. Chunk
    chunks = chunk_documents(documents)

    # 3. Embed
    embedder = get_embedder()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # 4. Store in Neo4j
    Neo4jDatabase.upsert_document(doc_id, path.name, len(chunks))

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}_chunk_{i}"
        Neo4jDatabase.upsert_chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=chunk.page_content,
            embedding=embedding,
            metadata=chunk.metadata,
            chunk_index=i,
        )

    # 5. Link consecutive chunks as a graph
    Neo4jDatabase.link_consecutive_chunks(doc_id)

    logger.info(
        "✅ Ingestion complete: {} chunks stored for '{}'", len(chunks), path.name
    )
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "total_chunks": len(chunks),
        "status": "success",
    }


# ── Query ──────────────────────────────────────────────────────────────────


class RAGChain:
    """End-to-end RAG: retrieve → format context → generate answer."""

    def __init__(self):
        self.retriever = Neo4jRetriever()
        self.chain = build_rag_chain()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline and return structured response.
        """
        # 1. Retrieve
        retrieved = self.retriever.retrieve(question)

        # 2. Format context
        context = format_context(retrieved)

        # 3. Generate
        answer = self.chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "sources": [
                {
                    "source": r.get("source"),
                    "page": r.get("page"),
                    "score": round(r.get("score", 0.0), 4),
                    "snippet": r.get("text", "")[:200] + "...",
                }
                for r in retrieved
                if r.get("score", 0.0) > 0  # only scored (non-neighbor) chunks
            ],
            "num_sources": len(retrieved),
        }

    def stream_query(self, question: str):
        """Stream tokens from the LLM (generator)."""
        retrieved = self.retriever.retrieve(question)
        context = format_context(retrieved)

        from ollama_model import get_llm, RAG_PROMPT

        llm = get_llm()
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        yield from llm.stream(prompt_text)
