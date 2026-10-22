
from unittest.mock import Mock

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from travel_agent.llm import model as model_module


@tool
def estimate_trip_cost(
    destination: str,
    nights: int,
    travellers: int = 1,
) -> dict:
    """Estimate accommodation costs for a trip."""
    return {
        "destination": destination,
        "nights": nights,
        "travellers": travellers,
    }


def test_model_can_bind_tools(monkeypatch):
    fake_model = Mock()
    fake_model.bind_tools.return_value = fake_model

    monkeypatch.setattr(
        model_module,
        "create_chat_model",
        lambda: fake_model,
    )

    bound_model = model_module.create_chat_model().bind_tools(
        [estimate_trip_cost]
    )

    assert bound_model is fake_model
    fake_model.bind_tools.assert_called_once_with([estimate_trip_cost])


def test_ai_message_can_represent_a_tool_call():
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "estimate_trip_cost",
                "args": {
                    "destination": "Aswan",
                    "nights": 3,
                    "travellers": 2,
                },
                "id": "call_test_1",
            }
        ],
    )

    assert message.tool_calls[0]["name"] == "estimate_trip_cost"
    assert message.tool_calls[0]["args"]["destination"] == "Aswan"
