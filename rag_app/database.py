"""
database.py — Neo4j connection manager with vector index support.
Implements a singleton driver + helper methods for the RAG pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Optional

from loguru import logger
from neo4j import GraphDatabase, Driver, Session
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

settings = get_settings()


class Neo4jDatabase:
    """Singleton Neo4j database manager."""

    _driver: Optional[Driver] = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    @classmethod
    def get_driver(cls) -> Driver:
        if cls._driver is None:
            cls._driver = cls._create_driver()
        return cls._driver

    @classmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _create_driver(cls) -> Driver:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        driver.verify_connectivity()
        logger.info("✅ Connected to Neo4j at {}", settings.neo4j_uri)
        return driver

    @classmethod
    def close(cls) -> None:
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j connection closed.")

    # ── context manager ────────────────────────────────────────────────────
    @classmethod
    @contextmanager
    def get_session(cls) -> Generator[Session, None, None]:
        driver = cls.get_driver()
        with driver.session() as session:
            yield session

    # ── schema setup ───────────────────────────────────────────────────────
    @classmethod
    def setup_schema(cls) -> None:
        """Create constraints and vector index if they don't exist."""
        with cls.get_session() as session:
            # Unique constraint on chunk id
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
            # Unique constraint on document id
            session.run(
                "CREATE CONSTRAINT doc_id IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
            )
            # Vector index for similarity search (Neo4j 5.11+)
            session.run(
                """
                CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 768,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
            )
        logger.info("✅ Neo4j schema & vector index ready.")

    # ── document operations ────────────────────────────────────────────────
    @classmethod
    def upsert_document(cls, doc_id: str, filename: str, total_chunks: int) -> None:
        with cls.get_session() as session:
            session.run(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.filename = $filename,
                    d.total_chunks = $total_chunks,
                    d.created_at = datetime()
                """,
                doc_id=doc_id,
                filename=filename,
                total_chunks=total_chunks,
            )

    @classmethod
    def upsert_chunk(
        cls,
        doc_id: str,
        chunk_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        chunk_index: int,
    ) -> None:
        with cls.get_session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text      = $text,
                    c.embedding = $embedding,
                    c.page       = $page,
                    c.chunk_index = $chunk_index,
                    c.source     = $source
                MERGE (d)-[:HAS_CHUNK {index: $chunk_index}]->(c)
                """,
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                page=metadata.get("page", 0),
                chunk_index=chunk_index,
                source=metadata.get("source", ""),
            )

    @classmethod
    def link_consecutive_chunks(cls, doc_id: str) -> None:
        """Create NEXT relationships between consecutive chunks."""
        with cls.get_session() as session:
            session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})-[r:HAS_CHUNK]->(c:Chunk)
                WITH c ORDER BY r.index
                WITH collect(c) AS chunks
                UNWIND range(0, size(chunks)-2) AS i
                WITH chunks[i] AS c1, chunks[i+1] AS c2
                MERGE (c1)-[:NEXT]->(c2)
                """,
                doc_id=doc_id,
            )

    # ── vector similarity search ───────────────────────────────────────────
    @classmethod
    def vector_search(
        cls,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        with cls.get_session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes(
                    'chunk_embedding', $top_k, $embedding
                ) YIELD node AS chunk, score
                MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
                RETURN chunk.chunk_id   AS chunk_id,
                       chunk.text       AS text,
                       chunk.page       AS page,
                       d.filename       AS source,
                       score
                ORDER BY score DESC
                """,
                embedding=query_embedding,
                top_k=top_k,
            )
            return [dict(r) for r in result]

    # ── list documents ─────────────────────────────────────────────────────
    @classmethod
    def list_documents(cls) -> List[Dict[str, Any]]:
        with cls.get_session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                RETURN d.doc_id AS doc_id,
                       d.filename AS filename,
                       d.total_chunks AS total_chunks,
                       d.created_at AS created_at
                ORDER BY d.created_at DESC
                """
            )
            return [dict(r) for r in result]

    @classmethod
    def delete_document(cls, doc_id: str) -> bool:
        with cls.get_session() as session:
            result = session.run(
                """
                MATCH (d:Document {doc_id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
                RETURN count(d) AS deleted
                """,
                doc_id=doc_id,
            )
            return result.single()["deleted"] > 0
