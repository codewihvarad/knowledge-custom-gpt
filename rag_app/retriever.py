"""
retriever.py — Neo4j-powered vector retriever for the RAG pipeline.
Performs hybrid retrieval: vector similarity + graph traversal.
"""
from typing import List, Dict, Any

from loguru import logger

from config import get_settings
from database import Neo4jDatabase
from embedding import get_embedder

settings = get_settings()


class Neo4jRetriever:
    """
    Retriever that:
    1. Embeds the user query with Ollama.
    2. Runs a cosine-similarity vector search over Neo4j Chunk nodes.
    3. Optionally fetches adjacent chunks via NEXT relationships (context window).
    """

    def __init__(
        self,
        top_k: int = settings.top_k_results,
        fetch_neighbors: bool = True,
    ):
        self.top_k = top_k
        self.fetch_neighbors = fetch_neighbors
        self.embedder = get_embedder()

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        logger.info("Retrieving context for query: '{}'", query[:80])

        # 1. Embed the query
        query_embedding = self.embedder.embed_query(query)

        # 2. Vector similarity search
        results = Neo4jDatabase.vector_search(query_embedding, top_k=self.top_k)

        if not results:
            logger.warning("No chunks found for query.")
            return []

        # 3. Graph context expansion — add previous/next chunks
        if self.fetch_neighbors:
            results = self._expand_with_neighbors(results)

        logger.info("Retrieved {} chunk(s) from Neo4j.", len(results))
        return results

    def _expand_with_neighbors(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fetch adjacent chunks via NEXT/PREVIOUS graph relationships."""
        seen_ids = {r["chunk_id"] for r in results}
        expanded = list(results)

        with Neo4jDatabase.get_session() as session:
            for doc in results:
                cid = doc["chunk_id"]
                neighbors = session.run(
                    """
                    MATCH (c:Chunk {chunk_id: $cid})
                    OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
                    OPTIONAL MATCH (c)-[:NEXT]->(nxt:Chunk)
                    MATCH (d:Document)-[:HAS_CHUNK]->(c)
                    RETURN
                        prev.chunk_id AS prev_id, prev.text AS prev_text,
                        prev.page     AS prev_page,
                        nxt.chunk_id  AS nxt_id,  nxt.text  AS nxt_text,
                        nxt.page      AS nxt_page,
                        d.filename    AS source
                    """,
                    cid=cid,
                ).single()

                if neighbors is None:
                    continue

                source = neighbors["source"] or doc.get("source", "")

                for prefix in ("prev", "nxt"):
                    n_id = neighbors[f"{prefix}_id"]
                    n_text = neighbors[f"{prefix}_text"]
                    n_page = neighbors[f"{prefix}_page"]
                    if n_id and n_id not in seen_ids and n_text:
                        seen_ids.add(n_id)
                        expanded.append(
                            {
                                "chunk_id": n_id,
                                "text": n_text,
                                "page": n_page,
                                "source": source,
                                "score": 0.0,  # neighbor, not directly scored
                            }
                        )

        return expanded
