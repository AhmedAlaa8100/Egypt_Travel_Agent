
from uuid import uuid4

from travel_agent.config import get_settings
from travel_agent.memory.store import MemoryStore


def test_memory_persists_across_store_instances():
    # Use a unique collection so this test does not pollute real memories.
    collection_name = f"test_memory_{uuid4().hex}"

    settings = get_settings()
    settings.chroma_persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_store = MemoryStore.__new__(MemoryStore)

    # Initialize a dedicated persistent collection for this test.
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_directory)
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(settings.embedding_model)

    first_store._client = client
    first_store._collection = collection
    first_store._embedder = embedder

    fact = f"I prefer window seats for flights. {uuid4().hex}"
    first_store.remember(fact, {"type": "test"})

    # Simulate reopening the same collection.
    second_store = MemoryStore.__new__(MemoryStore)
    second_store._client = client
    second_store._collection = client.get_collection(collection_name)
    second_store._embedder = embedder

    results = second_store.recall(fact, top_k=1)

    assert results
    assert results[0] == fact

    client.delete_collection(collection_name)
