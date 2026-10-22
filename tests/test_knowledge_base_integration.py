
import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from travel_agent.config import get_settings
from travel_agent.rag.store import KnowledgeBase


def test_real_semantic_retrieval():
    settings = get_settings()
    client = chromadb.Client()
    collection_name = "travel_rag_integration_test"

    collection = client.create_collection(name=collection_name)

    try:
        embedder = SentenceTransformer(settings.embedding_model)
        knowledge_base = KnowledgeBase(
            collection=collection,
            embedder=embedder,
        )

        knowledge_base.seed_default_documents()

        results = knowledge_base.search(
            "Where should I go in Egypt during winter?",
            top_k=3,
        )

        assert len(results) == 3
        assert "Alexandria" in results[0]["text"]
        assert results[0]["distance"] is not None

    finally:
        client.delete_collection(name=collection_name)
