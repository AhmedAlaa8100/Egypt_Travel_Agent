
from travel_agent.rag.store import KnowledgeBase
from travel_agent.tools.preference_tools import PreferenceTools
from travel_agent.tools.registry import create_tools


class FakeMemory:
    def remember(self, fact, metadata=None):
        pass

    def recall(self, query, top_k=3):
        return []


class FakeKnowledgeBase:
    def search(self, query, top_k=3):
        assert query == "winter trip"
        assert top_k == 3
        return [
            {
                "text": "Alexandria can be suitable for a winter trip.",
                "metadata": {"topic": "winter"},
                "distance": 0.15,
            }
        ]


def test_search_knowledge_base_tool():
    preferences = PreferenceTools(memory=FakeMemory())
    knowledge_base = FakeKnowledgeBase()

    tools = create_tools(
        preferences,
        knowledge_base=knowledge_base,
    )
    rag_tool = next(
        tool for tool in tools
        if tool.name == "search_knowledge_base"
    )

    result = rag_tool.invoke({"query": "winter trip"})

    assert len(result["documents"]) == 1
    assert "Alexandria" in result["documents"][0]["text"]
    assert result["documents"][0]["metadata"]["topic"] == "winter"
