
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from travel_agent.agent import TravelAgent


def test_agent_executes_tool_and_returns_final_answer():
    preferences = Mock()
    preferences.save_preference.return_value = {
        "saved": "I prefer quiet destinations"
    }

    fake_model = Mock()
    fake_model.bind_tools.return_value = fake_model

    fake_model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I prefer quiet destinations"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="I've saved your preference."),
    ]

    agent = TravelAgent(
        preferences=preferences,
        model=fake_model,
        max_iterations=3,
    )

    result = agent.run("Remember that I prefer quiet destinations.")

    assert result == "I've saved your preference."
    preferences.save_preference.assert_called_once_with(
        "I prefer quiet destinations"
    )
    assert fake_model.invoke.call_count == 2


def test_agent_rejects_empty_messages():
    agent = TravelAgent(
        preferences=Mock(),
        model=Mock(),
        max_iterations=2,
    )

    try:
        agent.run("   ")
    except ValueError as exc:
        assert "non-empty string" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an empty message")

def test_agent_returns_tool_failure_to_model(monkeypatch):
    from langchain_core.messages import ToolMessage

    import travel_agent.agent as agent_module

    preferences = Mock()
    fake_model = Mock()
    fake_model.bind_tools.return_value = fake_model

    fake_model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_preference",
                    "args": {"fact": "I like quiet destinations"},
                    "id": "call_failure_1",
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

    agent = TravelAgent(
        preferences=preferences,
        model=fake_model,
        max_iterations=3,
    )

    answer = agent.run("Remember that I like quiet destinations.")

    assert answer == "I couldn't save that preference."

    second_call_messages = fake_model.invoke.call_args_list[1].args[0]
    tool_messages = [
        message
        for message in second_call_messages
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 1
    assert "Temporary tool failure" in tool_messages[0].content
    preferences.save_preference.assert_not_called()
