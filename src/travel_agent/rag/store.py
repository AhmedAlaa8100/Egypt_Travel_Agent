
"""Persistent document storage and semantic retrieval for travel knowledge."""

from typing import Any, Sequence

import chromadb
from sentence_transformers import SentenceTransformer

from travel_agent.config import get_settings


DEFAULT_DOCUMENTS = {
    "alexandria-winter": (
        "Alexandria can be a suitable destination for a winter trip, "
        "especially for travellers who enjoy the Mediterranean coast."
    ),
    "cairo-city": (
        "Cairo is a large, busy city with historical landmarks, museums, "
        "markets, and cultural attractions throughout the year."
    ),
    "luxor-aswan-cruise": (
        "A Nile cruise between Luxor and Aswan usually takes "
        "3 to 4 nights."
    ),
}


class KnowledgeBase:
    """Store travel documents persistently and retrieve them by meaning."""

    def __init__(
        self,
        *,
        collection: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        settings = get_settings()

        if collection is None:
            client = chromadb.PersistentClient(
                path=str(settings.chroma_persist_directory)
            )
            collection = client.get_or_create_collection(
                name=settings.document_collection
            )

        self._collection = collection
        self._embedder = embedder or SentenceTransformer(
            settings.embedding_model
        )

    def add_documents(
        self,
        documents: Sequence[str],
        *,
        ids: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update documents using stable IDs."""
        if not documents:
            raise ValueError("documents cannot be empty")

        if len(documents) != len(ids):
            raise ValueError("documents and ids must have equal lengths")

        if len(set(ids)) != len(ids):
            raise ValueError("document IDs must be unique")

        cleaned = [document.strip() for document in documents]

        if any(not document for document in cleaned):
            raise ValueError("documents cannot contain empty text")

        if metadatas is not None and len(metadatas) != len(cleaned):
            raise ValueError("metadatas and documents must have equal lengths")

        embeddings = self._embedder.encode(cleaned)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        payload = {
            "ids": list(ids),
            "documents": cleaned,
            "embeddings": embeddings,
        }
        if metadatas is not None:
            payload["metadatas"] = list(metadatas)

        self._collection.upsert(**payload)

    def seed_default_documents(self) -> None:
        """Add the built-in travel documents without duplicating them."""
        self.add_documents(
            list(DEFAULT_DOCUMENTS.values()),
            ids=list(DEFAULT_DOCUMENTS.keys()),
            metadatas=[
                {"topic": "winter"},
                {"topic": "destination"},
                {"topic": "cruise"},
            ],
        )

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the most relevant documents for a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self._embedder.encode([query.strip()])
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        return [
            {
                "text": document,
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distances[index] if index < len(distances) else None,
            }
            for index, document in enumerate(documents)
        ]
