
from unittest.mock import Mock

import pytest
from langchain_core.tools import tool

from travel_agent.tools.execution import invoke_tool_with_retry


@tool
def example_tool(value: str) -> str:
    """Return the supplied value."""
    return value


def test_successful_tool_runs_once():
    result, attempts = invoke_tool_with_retry(
        example_tool,
        {"value": "hello"},
        sleep_fn=Mock(),
    )

    assert result == "hello"
    assert attempts == 1


def test_transient_error_is_retried():
    fake_tool = Mock()
    fake_tool.invoke.side_effect = [
        TimeoutError("temporary failure"),
        "success",
    ]
    fake_sleep = Mock()

    result, attempts = invoke_tool_with_retry(
        fake_tool,
        {},
        max_retries=2,
        base_delay_seconds=0.25,
        sleep_fn=fake_sleep,
    )

    assert result == "success"
    assert attempts == 2
    fake_sleep.assert_called_once_with(0.25)


def test_invalid_input_is_not_retried():
    fake_tool = Mock()
    fake_tool.invoke.side_effect = ValueError("invalid destination")

    with pytest.raises(ValueError, match="invalid destination"):
        invoke_tool_with_retry(
            fake_tool,
            {},
            max_retries=3,
            sleep_fn=Mock(),
        )

    assert fake_tool.invoke.call_count == 1


def test_exhausted_retries_raise_last_error():
    fake_tool = Mock()
    fake_tool.invoke.side_effect = TimeoutError("still unavailable")
    fake_sleep = Mock()

    with pytest.raises(TimeoutError, match="still unavailable"):
        invoke_tool_with_retry(
            fake_tool,
            {},
            max_retries=2,
            sleep_fn=fake_sleep,
        )

    assert fake_tool.invoke.call_count == 3
    assert fake_sleep.call_count == 2
