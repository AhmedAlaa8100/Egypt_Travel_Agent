from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

from travel_agent.config import get_settings


class MemoryStore:
    """Stores facts and retrieves them by semantic similarity."""

    def __init__(self) -> None:
        settings = get_settings()

        # Create the directory if it does not exist.
        settings.chroma_persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # PersistentClient saves data to disk across program restarts.
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_directory)
        )

        # Reuse the collection instead of deleting saved memories.
        self._collection = self._client.get_or_create_collection(
            name=settings.memory_collection,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedder = SentenceTransformer(
            settings.embedding_model
        )

    def remember(self, fact_text: str, metadata: dict | None = None) -> str:
        """Save a fact and return its unique memory ID."""

        if not isinstance(fact_text, str):
            raise TypeError("fact_text must be a string")

        fact_text = fact_text.strip()
        if not fact_text:
            raise ValueError("fact_text cannot be empty")

        memory_id = f"mem_{uuid4().hex}"
        embedding = self._embedder.encode(fact_text).tolist()

        self._collection.add(
            ids=[memory_id],
            documents=[fact_text],
            embeddings=[embedding],
            metadatas=[
                metadata or {"type": "agent_generated"}
            ],
        )

        return memory_id

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """Return the most semantically relevant saved facts."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self._embedder.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )

        return results["documents"][0] or []
