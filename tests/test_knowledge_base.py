
import pytest

from travel_agent.rag.store import KnowledgeBase, DEFAULT_DOCUMENTS


class FakeEmbedder:
    def encode(self, texts):
        # Deterministic vectors; no model download required.
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    def __init__(self):
        self.documents = {}

    def upsert(self, ids, documents, embeddings, metadatas=None):
        metadatas = metadatas or [{} for _ in ids]
        for i, document_id in enumerate(ids):
            self.documents[document_id] = {
                "document": documents[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i],
            }

    def count(self):
        return len(self.documents)

    def query(self, query_embeddings, n_results, include):
        items = list(self.documents.items())[:n_results]
        return {
            "documents": [[item["document"] for _, item in items]],
            "metadatas": [[item["metadata"] for _, item in items]],
            "distances": [[0.1 for _ in items]],
        }


@pytest.fixture
def knowledge_base():
    return KnowledgeBase(
        collection=FakeCollection(),
        embedder=FakeEmbedder(),
    )


def test_add_and_search_documents(knowledge_base):
    knowledge_base.add_documents(
        ["Alexandria is a coastal city."],
        ids=["alexandria"],
    )

    results = knowledge_base.search("coastal destination")

    assert len(results) == 1
    assert results[0]["text"] == "Alexandria is a coastal city."
    assert "distance" in results[0]


def test_seed_default_documents(knowledge_base):
    knowledge_base.seed_default_documents()
    knowledge_base.seed_default_documents()

    assert knowledge_base._collection.count() == len(DEFAULT_DOCUMENTS)


def test_empty_collection_returns_no_results(knowledge_base):
    assert knowledge_base.search("winter trip") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(knowledge_base, query):
    with pytest.raises(ValueError):
        knowledge_base.search(query)


def test_search_rejects_invalid_top_k(knowledge_base):
    with pytest.raises(ValueError):
        knowledge_base.search("winter", top_k=0)


def test_add_documents_rejects_mismatched_ids(knowledge_base):
    with pytest.raises(ValueError):
        knowledge_base.add_documents(
            ["One document", "Another document"],
            ids=["only-one-id"],
        )
