
import json
from unittest.mock import Mock

from langchain_core.messages import AIMessage, ToolMessage

from travel_agent.agent import TravelAgent
from travel_agent.tools.preference_tools import PreferenceTools


class FakeMemory:
    def remember(self, fact, metadata=None):
        pass

    def recall(self, query, top_k=3):
        return []


class FakeKnowledgeBase:
    def __init__(self):
        self.search_queries = []

    def seed_default_documents(self):
        pass

    def search(self, query, top_k=3):
        self.search_queries.append(query)
        return [
            {
                "text": "Alexandria can be suitable for a winter trip.",
                "metadata": {"topic": "winter"},
                "distance": 0.15,
            }
        ]


def test_agent_uses_retrieved_documents_before_answering():
    knowledge_base = FakeKnowledgeBase()

    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "winter trip"},
                    "id": "rag_call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="Alexandria can be a good choice for a winter trip."
        ),
    ]

    agent = TravelAgent(
        preferences=PreferenceTools(memory=FakeMemory()),
        model=model,
        knowledge_base=knowledge_base,
    )

    answer = agent.run(
        "Where should I go in Egypt during winter?",
        thread_id="rag-test",
    )

    assert "Alexandria" in answer
    assert knowledge_base.search_queries == ["winter trip"]
    assert model.invoke.call_count == 2

    second_call = model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "rag_call_1"

    payload = json.loads(tool_messages[0].content)
    assert payload["success"] is True
    assert (
        "Alexandria"
        in payload["result"]["documents"][0]["text"]
    )


def test_rag_thread_does_not_leak_into_another_thread():
    knowledge_base = FakeKnowledgeBase()

    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(content="Alexandria is worth considering."),
        AIMessage(content="What would you like to know?"),
    ]

    agent = TravelAgent(
        preferences=PreferenceTools(memory=FakeMemory()),
        model=model,
        knowledge_base=knowledge_base,
    )

    agent.run(
        "Tell me about Alexandria.",
        thread_id="rag-thread-a",
    )
    agent.run(
        "Hello.",
        thread_id="rag-thread-b",
    )

    second_call = model.invoke.call_args_list[1].args[0]
    user_messages = [
        message.content
        for message in second_call
        if message.type == "human"
    ]

    assert user_messages == ["Hello."]
