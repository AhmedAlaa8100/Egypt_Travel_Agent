
"""Reliable execution for LangChain tools."""

from time import sleep
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from travel_agent.config import get_settings


RETRYABLE_ERRORS = (TimeoutError, ConnectionError)
INVALID_INPUT_ERRORS = (
    ValueError,
    TypeError,
    SyntaxError,
    ValidationError,
)


def invoke_tool_with_retry(
    tool: BaseTool,
    args: dict[str, Any],
    *,
    max_retries: int | None = None,
    base_delay_seconds: float = 0.25,
    sleep_fn: Callable[[float], None] = sleep,
) -> tuple[Any, int]:
    """Invoke a tool, retrying transient failures with exponential backoff.

    max_retries counts retries after the initial attempt.
    Returns (tool_result, total_attempts).
    Invalid input and unexpected errors are not retried.
    """
    if max_retries is None:
        max_retries = get_settings().tool_max_retries

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")

    if base_delay_seconds < 0:
        raise ValueError("base_delay_seconds cannot be negative")

    for attempt in range(max_retries + 1):
        try:
            result = tool.invoke(args)
            return result, attempt + 1

        except INVALID_INPUT_ERRORS:
            raise

        except RETRYABLE_ERRORS:
            if attempt == max_retries:
                raise

            delay = base_delay_seconds * (2**attempt)
            sleep_fn(delay)

    raise RuntimeError("Unexpected tool execution state")
