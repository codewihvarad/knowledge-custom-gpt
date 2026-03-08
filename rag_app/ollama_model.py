"""
ollama_model.py — Production Ollama LLM wrapper with streaming support.
"""
from functools import lru_cache
from typing import Generator, Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from config import get_settings

settings = get_settings()

# ── RAG system prompt ──────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert AI assistant with deep knowledge extracted from the provided documents.

STRICT RULES:
1. Answer ONLY based on the provided context below.
2. If the context doesn't contain the answer, say: "I don't have enough information in the provided documents to answer this question."
3. Be concise, structured, and professional.
4. Always cite the source page/document when referencing specific facts.
5. Use markdown formatting for better readability.

CONTEXT:
{context}

---
Question: {question}

Answer (cite sources where applicable):"""

RAG_PROMPT = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)


@lru_cache
def get_llm() -> OllamaLLM:
    """Return a cached OllamaLLM singleton."""
    llm = OllamaLLM(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=2048,   # keep RAM usage low for small models
    )
    logger.info("✅ Ollama LLM ready (model={})", settings.ollama_model)
    return llm


def build_rag_chain():
    """Return a LangChain LCEL chain: prompt | llm | output_parser."""
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain


def format_context(retrieved_docs: list) -> str:
    """Format retrieved Neo4j results into a numbered context block."""
    if not retrieved_docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("source", "unknown")
        page = doc.get("page", "?")
        score = doc.get("score", 0.0)
        text = doc.get("text", "")
        parts.append(
            f"[{i}] Source: {source} | Page: {page} | Score: {score:.3f}\n{text}"
        )

    return "\n\n".join(parts)