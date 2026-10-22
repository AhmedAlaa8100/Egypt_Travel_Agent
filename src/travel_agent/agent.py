
"""Run the travel agent's tool-calling loop."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from travel_agent.config import get_settings
from travel_agent.llm.model import create_chat_model
from travel_agent.memory.store import MemoryStore
from travel_agent.tools.preference_tools import PreferenceTools
from travel_agent.tools.registry import create_tools
from travel_agent.rag.store import KnowledgeBase

from travel_agent.tools.execution import (
    INVALID_INPUT_ERRORS,
    RETRYABLE_ERRORS,
    invoke_tool_with_retry,
)

class TravelAgent:
    """Coordinate the LLM and tools to answer travel-related requests."""

    def __init__(
        self,
        preferences: PreferenceTools | None = None,
        model: BaseChatModel | None = None,
        max_iterations: int | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ) -> None:
        self._preferences = preferences or PreferenceTools(MemoryStore())
        self._knowledge_base = knowledge_base or KnowledgeBase()
        self._knowledge_base.seed_default_documents()

        self._tools = create_tools(
            self._preferences,
            self._knowledge_base,
        )
        self._tools_by_name = {tool.name: tool for tool in self._tools}

        base_model = model or create_chat_model()
        self._model = base_model.bind_tools(self._tools)

        settings = get_settings()
        self._max_iterations = (
            max_iterations
            if max_iterations is not None
            else settings.agent_max_iterations
        )
        
    def run(self, user_message: str) -> str:
        """Answer a user message, executing tool calls when requested."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        messages = [
            SystemMessage(
                content=(
                    "You are a helpful travel assistant. "
                    "Use the available tools when they help answer the user. "
                    "Do not invent tool results. Explain estimates clearly."
                    "Treat tool errors as failures, not successful results. "
                    "If a tool fails, correct invalid arguments or try another "
                    "reasonable approach when possible. Never claim an operation "
                    "succeeded unless the tool result confirms it."
                )
            ),
            HumanMessage(content=user_message.strip()),
        ]

        for _ in range(self._max_iterations):
            response = self._model.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                if isinstance(response.content, str):
                    return response.content
                return str(response.content)

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                if tool_name not in self._tools_by_name:
                    raise ValueError(f"Unknown tool requested: {tool_name}")

                try:
                    result, attempts = invoke_tool_with_retry(
                        self._tools_by_name[tool_name],
                        tool_args,
                    )
                    tool_result = {
                        "success": True,
                        "result": result,
                        "attempts": attempts,
                    }

                except RETRYABLE_ERRORS as exc:
                    tool_result = {
                        "success": False,
                        "error": (
                            f"Temporary tool failure after retries: {exc}"
                        ),
                        "retryable": True,
                    }

                except INVALID_INPUT_ERRORS as exc:
                    tool_result = {
                        "success": False,
                        "error": f"Invalid tool input: {exc}",
                        "retryable": False,
                    }

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result, default=str),
                        tool_call_id=tool_call_id,
                    )
                )

        raise RuntimeError(
            f"Agent reached its limit of {self._max_iterations} iterations "
            "without producing a final answer."
        )
