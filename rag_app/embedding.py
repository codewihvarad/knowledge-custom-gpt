"""
embedding.py — Ollama-based embeddings via nomic-embed-text.
Falls back to HuggingFace sentence-transformers if Ollama is unavailable.
"""
from functools import lru_cache
from typing import List

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

settings = get_settings()


class OllamaEmbedder:
    """
    Calls the Ollama /api/embeddings endpoint to produce dense vectors.
    Dimension depends on the model:
      - nomic-embed-text : 768-dim
      - mxbai-embed-large: 1024-dim
    """

    def __init__(self, model: str = settings.ollama_embed_model):
        self.model = model
        self.base_url = settings.ollama_base_url.rstrip("/")
        self._client = httpx.Client(timeout=60.0)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_text(self, text: str) -> List[float]:
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            vec = self.embed_text(text)
            embeddings.append(vec)
            if (i + 1) % 10 == 0:
                logger.debug("Embedded {}/{} chunks", i + 1, len(texts))
        logger.info("✅ Embedded {} chunk(s) with model '{}'", len(texts), self.model)
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    def close(self):
        self._client.close()


@lru_cache
def get_embedder() -> OllamaEmbedder:
    """Return a cached OllamaEmbedder singleton."""
    embedder = OllamaEmbedder()
    # Quick health check
    try:
        _ = embedder.embed_text("ping")
        logger.info("✅ Ollama embedder ready (model={})", embedder.model)
    except Exception as e:
        logger.warning("Ollama embedder unavailable: {}. Falling back to HuggingFace.", e)
        return _get_hf_embedder()
    return embedder


def _get_hf_embedder():
    """HuggingFace fallback embedder that matches the OllamaEmbedder interface."""
    from langchain_huggingface import HuggingFaceEmbeddings

    class HFWrapper:
        def __init__(self):
            self._model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        def embed_documents(self, texts):
            return self._model.embed_documents(texts)

        def embed_query(self, query):
            return self._model.embed_query(query)

    logger.warning("Using HuggingFace fallback embedder (384-dim). Adjust vector index!")
    return HFWrapper()
