
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from travel_agent.agent import LIMIT_MESSAGE, TravelAgent

from langchain_core.messages import SystemMessage

class FakeKnowledgeBase:
    def seed_default_documents(self):
        pass

    def search(self, query, top_k=3):
        return [
            {
                "text": "Alexandria is suitable for a winter trip.",
                "metadata": {"topic": "winter"},
                "distance": 0.15,
            }
        ]


class FakeMemory:
    def remember(self, fact, metadata=None):
        pass

    def recall(self, query, top_k=3):
        return []


def make_agent(
    model=None,
    preferences=None,
    max_iterations=5,
):
    model = model or Mock()
    model.bind_tools.return_value = model

    return TravelAgent(
        preferences=preferences or Mock(),
        model=model,
        max_iterations=max_iterations,
        knowledge_base=FakeKnowledgeBase(),
    )


def test_agent_returns_direct_model_answer():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="Alexandria is a good winter destination."
    )

    agent = make_agent(model=model)

    answer = agent.run(
        "Where should I go in winter?",
        thread_id="direct-answer",
    )

    assert answer == "Alexandria is a good winter destination."
    assert model.invoke.call_count == 1


def test_agent_executes_tool_and_returns_final_answer():
    preferences = Mock()
    preferences.save_preference.return_value = {
        "saved": "I prefer quiet destinations"
    }

    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I prefer quiet destinations"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="I've saved your preference."),
    ]

    agent = make_agent(model=model, preferences=preferences)

    answer = agent.run(
        "Remember that I prefer quiet destinations.",
        thread_id="save-preference",
    )

    assert answer == "I've saved your preference."
    preferences.save_preference.assert_called_once_with(
        "I prefer quiet destinations"
    )
    assert model.invoke.call_count == 2


def test_agent_rejects_empty_user_message():
    agent = make_agent()

    with pytest.raises(ValueError, match="non-empty string"):
        agent.run("   ", thread_id="empty-message")


@pytest.mark.parametrize("thread_id", ["", "   "])
def test_agent_rejects_empty_thread_id(thread_id):
    agent = make_agent()

    with pytest.raises(ValueError, match="thread_id"):
        agent.run("Hello", thread_id=thread_id)


def test_agent_rejects_non_string_user_message():
    agent = make_agent()

    with pytest.raises(ValueError, match="non-empty string"):
        agent.run(None, thread_id="invalid-message")


def test_agent_preserves_history_without_duplicate_messages():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(content="I can help plan that trip."),
        AIMessage(content="You said you prefer quiet destinations."),
    ]

    agent = make_agent(model=model)
    thread_id = "continuity-test"

    agent.run(
        "I prefer quiet destinations.",
        thread_id=thread_id,
    )
    answer = agent.run(
        "What kind of destinations do I prefer?",
        thread_id=thread_id,
    )

    assert answer == "You said you prefer quiet destinations."

    second_call = model.invoke.call_args_list[1].args[0]
    human_messages = [
        message.content
        for message in second_call
        if isinstance(message, HumanMessage)
    ]

    assert human_messages == [
        "I prefer quiet destinations.",
        "What kind of destinations do I prefer?",
    ]
    assert len(human_messages) == len(set(human_messages))


def test_agent_isolates_different_threads():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(content="First conversation response."),
        AIMessage(content="Second conversation response."),
    ]

    agent = make_agent(model=model)

    agent.run("My destination is Alexandria.", thread_id="thread-a")
    agent.run("Hello.", thread_id="thread-b")

    second_call = model.invoke.call_args_list[1].args[0]
    human_messages = [
        message.content
        for message in second_call
        if isinstance(message, HumanMessage)
    ]

    assert human_messages == ["Hello."]
    assert "My destination is Alexandria." not in human_messages


def test_agent_returns_tool_failure_to_model(monkeypatch):
    import travel_agent.agent as agent_module

    preferences = Mock()
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I like quiet destinations"},
                    "id": "call_failure_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="I couldn't save that preference."),
    ]

    def fail_tool(*args, **kwargs):
        raise TimeoutError("service unavailable")

    monkeypatch.setattr(
        agent_module,
        "invoke_tool_with_retry",
        fail_tool,
    )

    agent = make_agent(model=model, preferences=preferences)

    answer = agent.run(
        "Remember that I like quiet destinations.",
        thread_id="tool-failure",
    )

    assert answer == "I couldn't save that preference."

    second_call = model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_failure_1"

    payload = json.loads(tool_messages[0].content)
    assert payload["success"] is False
    assert "Temporary tool failure" in payload["error"]
    preferences.save_preference.assert_not_called()


def test_agent_handles_multiple_tool_calls(monkeypatch):
    import travel_agent.agent as agent_module

    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I like museums"},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "save_preference",
                    "args": {"fact": "I like quiet hotels"},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="I've processed both preferences."),
    ]

    executed = []

    def fake_invoke_tool(tool, tool_args):
        executed.append(tool_args["fact"])
        return {"saved": tool_args["fact"]}, 1

    monkeypatch.setattr(
        agent_module,
        "invoke_tool_with_retry",
        fake_invoke_tool,
    )

    agent = make_agent(model=model)

    answer = agent.run(
        "Save my travel preferences.",
        thread_id="multiple-tools",
    )

    assert answer == "I've processed both preferences."
    assert executed == ["I like museums", "I like quiet hotels"]

    second_call = model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 2
    assert {message.tool_call_id for message in tool_messages} == {
        "call_1",
        "call_2",
    }


def test_agent_handles_unknown_tool_call():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "nonexistent_tool",
                    "args": {},
                    "id": "unknown_call",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="I couldn't use that tool."),
    ]

    agent = make_agent(model=model)

    answer = agent.run(
        "Do something.",
        thread_id="unknown-tool",
    )

    assert answer == "I couldn't use that tool."

    second_call = model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "unknown_call"
    assert "Unknown tool" in tool_messages[0].content


def test_iteration_limit_returns_clear_answer_and_closes_tool_calls():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "save_preference",
                "args": {"fact": "I like museums"},
                "id": "limited_call",
                "type": "tool_call",
            }
        ],
    )

    agent = make_agent(model=model, max_iterations=1)

    answer = agent.run(
        "Save my preference.",
        thread_id="iteration-limit",
    )

    assert answer == LIMIT_MESSAGE
    assert model.invoke.call_count == 1

    state = agent._graph.get_state(
        {"configurable": {"thread_id": "iteration-limit"}}
    ).values

    messages = state["messages"]
    tool_messages = [
        message
        for message in messages
        if isinstance(message, ToolMessage)
    ]

    assert any(
        message.tool_call_id == "limited_call"
        for message in tool_messages
    )
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == LIMIT_MESSAGE



def test_agent_trims_history_sent_to_model():
    model = Mock()
    model.bind_tools.return_value = model
    model.invoke.side_effect = [
        AIMessage(content="First response."),
        AIMessage(content="Second response."),
    ]

    # Count each message as one token, so trimming is predictable.
    def count_tokens(messages):
        return len(messages)

    model.get_num_tokens_from_messages.side_effect = count_tokens

    agent = TravelAgent(
        preferences=Mock(),
        model=model,
        knowledge_base=FakeKnowledgeBase(),
        max_history_tokens=3,
    )

    agent.run("First user message.", thread_id="trim-test")
    agent.run("Second user message.", thread_id="trim-test")

    second_call = model.invoke.call_args_list[1].args[0]

    # The system prompt is added separately and appears exactly once.
    assert isinstance(second_call[0], SystemMessage)
    assert sum(
        isinstance(message, SystemMessage)
        for message in second_call
    ) == 1

    # The latest user message must survive trimming.
    assert any(
        isinstance(message, HumanMessage)
        and message.content == "Second user message."
        for message in second_call
    )


from itertools import count


def test_agent_stops_when_model_keeps_requesting_tools():
    model = Mock()
    model.bind_tools.return_value = model

    call_numbers = count(1)

    def keep_requesting_tool(*args, **kwargs):
        number = next(call_numbers)

        # Return a fresh AIMessage with unique IDs on every call.
        return AIMessage(
            id=f"ai_loop_{number}",
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I like museums"},
                    "id": f"loop_call_{number}",
                    "type": "tool_call",
                }
            ],
        )

    model.invoke.side_effect = keep_requesting_tool

    agent = make_agent(model=model, max_iterations=2)

    answer = agent.run(
        "Save my preference.",
        thread_id="iteration-loop",
    )

    assert answer == LIMIT_MESSAGE
    assert model.invoke.call_count == 2

    state = agent._graph.get_state(
        {"configurable": {"thread_id": "iteration-loop"}}
    ).values

    assert state["iterations"] == 2
    assert state["answer"] == LIMIT_MESSAGE
    assert isinstance(state["messages"][-1], AIMessage)
    assert state["messages"][-1].content == LIMIT_MESSAGE
