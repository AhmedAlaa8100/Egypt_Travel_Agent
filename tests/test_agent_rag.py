
import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from travel_agent.agent import TravelAgent
from travel_agent.tools.preference_tools import PreferenceTools


class FakeMemory:
    def remember(self, fact, metadata=None):
        pass

    def recall(self, query, top_k=3):
        return []


class FakeKnowledgeBase:
    def seed_default_documents(self):
        pass

    def search(self, query, top_k=3):
        return [
            {
                "text": "Alexandria can be suitable for a winter trip.",
                "metadata": {"topic": "winter"},
                "distance": 0.15,
            }
        ]


def test_agent_uses_retrieved_documents_before_answering():
    # Simulate the LLM requesting the RAG tool, then answering.
    model = MagicMock()
    bound_model = MagicMock()

    model.bind_tools.return_value = bound_model
    bound_model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "args": {"query": "winter trip"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="Alexandria can be a good choice for a winter trip.",
        ),
    ]

    agent = TravelAgent(
        preferences=PreferenceTools(memory=FakeMemory()),
        model=model,
        knowledge_base=FakeKnowledgeBase(),
    )

    answer = agent.run("Where should I go in Egypt during winter?")

    assert "Alexandria" in answer
    assert bound_model.invoke.call_count == 2

    # Verify the retrieved document was returned to the LLM.
    second_call_messages = bound_model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call_messages
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    tool_result = json.loads(tool_messages[0].content)

    assert tool_result["success"] is True
    assert "Alexandria" in tool_result["result"]["documents"][0]["text"]
