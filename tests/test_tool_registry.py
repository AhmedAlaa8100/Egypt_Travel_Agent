
from unittest.mock import Mock

import pytest

from tests.test_preference_tools import preferences
from travel_agent.tools.registry import create_tools


@pytest.fixture
def setup_tools():
    preferences = Mock()
    tools = create_tools(preferences)
    tools_by_name = {tool.name: tool for tool in tools}
    return preferences, tools_by_name


def test_creates_five_langchain_tools():
    # Keep your existing preferences setup.
    tools = create_tools(preferences)

    assert {tool.name for tool in tools} == {
        "estimate_trip_cost",
        "convert_currency",
        "save_preference",
        "recall_preferences",
        "search_knowledge_base",
    }


def test_tool_schemas_are_generated(setup_tools):
    _, tools = setup_tools

    schema = tools["estimate_trip_cost"].args_schema.model_json_schema()

    assert "destination" in schema["properties"]
    assert "nights" in schema["properties"]
    assert "travellers" in schema["properties"]


def test_save_preference_tool(setup_tools):
    preferences, tools = setup_tools
    preferences.save_preference.return_value = {
        "saved": "I prefer quiet destinations"
    }

    result = tools["save_preference"].invoke(
        {"fact": "I prefer quiet destinations"}
    )

    assert result == {"saved": "I prefer quiet destinations"}
    preferences.save_preference.assert_called_once_with(
        "I prefer quiet destinations"
    )


def test_recall_preferences_tool(setup_tools):
    preferences, tools = setup_tools
    preferences.recall_preferences.return_value = {
        "memories": ["I prefer quiet destinations"]
    }

    result = tools["recall_preferences"].invoke(
        {"query": "destination preferences"}
    )

    assert result == {
        "memories": ["I prefer quiet destinations"]
    }
    preferences.recall_preferences.assert_called_once_with(
        "destination preferences"
    )


def test_tool_rejects_missing_required_argument(setup_tools):
    _, tools = setup_tools

    with pytest.raises(Exception):
        tools["save_preference"].invoke({})
