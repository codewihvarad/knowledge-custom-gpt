"""
config.py — Centralised configuration using pydantic-settings.
All values are read from the .env file (or environment variables).
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Neo4j ──────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "12345678"

    # ── Ollama ─────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_embed_model: str = "nomic-embed-text"

    # ── App ────────────────────────────────────────────────────────────────
    app_name: str = "RAG Knowledge Bot"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── Retrieval ──────────────────────────────────────────────────────────
    top_k_results: int = 5
    similarity_threshold: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()