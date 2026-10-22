
from unittest.mock import Mock

import pytest

from travel_agent.tools.preference_tools import PreferenceTools


@pytest.fixture
def memory():
    return Mock()


@pytest.fixture
def preferences(memory):
    return PreferenceTools(memory=memory)


def test_save_preference(preferences, memory):
    result = preferences.save_preference(
        "I prefer quiet destinations"
    )

    assert result == {
        "saved": "I prefer quiet destinations"
    }
    memory.remember.assert_called_once_with(
        "I prefer quiet destinations",
        metadata={"type": "preference"},
    )


def test_recall_preferences(preferences, memory):
    memory.recall.return_value = [
        "I prefer quiet destinations"
    ]

    result = preferences.recall_preferences("destination preferences")

    assert result == {
        "memories": ["I prefer quiet destinations"]
    }
    memory.recall.assert_called_once_with(
        "destination preferences",
        top_k=3,
    )


def test_save_preference_rejects_short_text(preferences, memory):
    with pytest.raises(ValueError):
        preferences.save_preference("  ")

    memory.remember.assert_not_called()


def test_recall_preferences_rejects_empty_query(preferences, memory):
    with pytest.raises(ValueError):
        preferences.recall_preferences("  ")

    memory.recall.assert_not_called()
