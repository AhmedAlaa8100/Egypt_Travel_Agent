"""Run the travel agent using a LangGraph workflow."""

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, MessagesState, StateGraph

from travel_agent.config import get_settings
from travel_agent.llm.model import create_chat_model
from travel_agent.memory.store import MemoryStore
from travel_agent.rag.store import KnowledgeBase
from travel_agent.tools.execution import (
    INVALID_INPUT_ERRORS,
    RETRYABLE_ERRORS,
    invoke_tool_with_retry,
)
from travel_agent.tools.preference_tools import PreferenceTools
from travel_agent.tools.registry import create_tools


SYSTEM_PROMPT = (
    "You are a helpful travel assistant. "
    "Use the available tools when they help answer the user. "
    "Do not invent tool results. Explain estimates clearly. "
    "Treat tool errors as failures, not successful results. "
    "If a tool fails, correct invalid arguments or try another "
    "reasonable approach when possible. Never claim an operation "
    "succeeded unless the tool result confirms it."
)

LIMIT_MESSAGE = (
    "I reached the tool-call limit before completing your request. "
    "Please narrow your request or ask me to continue with a "
    "specific part."
)


class AgentState(MessagesState):
    """State shared by all nodes in the agent graph."""

    iterations: int
    answer: str


class TravelAgent:
    """Coordinate the LLM and tools through LangGraph."""

    def __init__(
        self,
        preferences: PreferenceTools | None = None,
        model: BaseChatModel | None = None,
        max_iterations: int | None = None,
        knowledge_base: KnowledgeBase | None = None,
        max_history_tokens: int | None = None,
    ) -> None:
        self._preferences = preferences or PreferenceTools(MemoryStore())
        self._knowledge_base = knowledge_base or KnowledgeBase()
        self._knowledge_base.seed_default_documents()

        self._tools = create_tools(
            self._preferences,
            self._knowledge_base,
        )
        self._tools_by_name = {tool.name: tool for tool in self._tools}

        # Keep the unbound model: trim_messages needs it for token counting.
        self._base_model = model or create_chat_model()
        self._model = self._base_model.bind_tools(self._tools)

        # None disables trimming; set a number to cap the history sent
        # to the model on each call.
        self._max_history_tokens = max_history_tokens

        settings = get_settings()
        self._max_iterations = (
            max_iterations
            if max_iterations is not None
            else settings.agent_max_iterations
        )

        workflow = StateGraph(AgentState)
        workflow.add_node("model", self._call_model)
        workflow.add_node("tools", self._execute_tools)
        workflow.add_node("limit", self._handle_limit)

        workflow.set_entry_point("model")
        workflow.add_conditional_edges(
            "model",
            self._route_after_model,
            {
                "tools": "tools",
                "limit": "limit",
                "end": END,
            },
        )
        workflow.add_edge("tools", "model")
        workflow.add_edge("limit", END)

        self._checkpointer = InMemorySaver()
        self._graph = workflow.compile(checkpointer=self._checkpointer)

    def _call_model(self, state: AgentState) -> dict:
        """Call the model without storing the system prompt in history."""
        history = state["messages"]

        if self._max_history_tokens is not None:
            history = trim_messages(
                history,
                max_tokens=self._max_history_tokens,
                token_counter=self._base_model,
                strategy="last",
                start_on="human",
                include_system=False,
            )

        messages = [SystemMessage(content=SYSTEM_PROMPT), *history]
        response = self._model.invoke(messages)

        return {
            "messages": [response],
            "iterations": state["iterations"] + 1,
        }


    def _route_after_model(self, state: AgentState) -> str:
        """Choose whether to execute tools, stop, or report the limit."""
        response = state["messages"][-1]

        # Routing should always inspect the latest model response.
        if not isinstance(response, AIMessage):
            raise TypeError(
                "Expected the latest message to be an AIMessage after "
                f"the model node, got {type(response).__name__}."
            )

        if not response.tool_calls:
            return "end"

        if state["iterations"] >= self._max_iterations:
            return "limit"

        return "tools"



    def _execute_tools(self, state: AgentState) -> dict:
        """Execute every tool call and return a result for each call."""
        response = state["messages"][-1]
        tool_messages = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_call_id = tool_call["id"]
            tool_args = tool_call["args"]

            if tool_name not in self._tools_by_name:
                tool_result = {
                    "success": False,
                    "error": f"Unknown tool requested: {tool_name}",
                    "retryable": False,
                }
            else:
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
                        "error": f"Temporary tool failure after retries: {exc}",
                        "retryable": True,
                    }

                except INVALID_INPUT_ERRORS as exc:
                    tool_result = {
                        "success": False,
                        "error": f"Invalid tool input: {exc}",
                        "retryable": False,
                    }

                except Exception as exc:
                    tool_result = {
                        "success": False,
                        "error": (
                            f"Unexpected tool failure: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        "retryable": False,
                    }

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, default=str),
                    tool_call_id=tool_call_id,
                )
            )

        return {"messages": tool_messages}

    def _handle_limit(self, state: AgentState) -> dict:
        """Report the iteration limit while keeping history valid.

        The last message is an AIMessage with tool calls that will never
        run. Because history now persists across turns, every tool call
        must be answered by a ToolMessage, otherwise the next request on
        this thread would be rejected by the model provider.
        """
        response = state["messages"][-1]

        skipped = [
            ToolMessage(
                content=json.dumps(
                    {
                        "success": False,
                        "error": "Not executed: iteration limit reached.",
                        "retryable": False,
                    }
                ),
                tool_call_id=tool_call["id"],
            )
            for tool_call in response.tool_calls
        ]

        return {
            "messages": [*skipped, AIMessage(content=LIMIT_MESSAGE)],
            "answer": LIMIT_MESSAGE,
        }

    def run(self, user_message: str, thread_id: str) -> str:
        """Answer a user message using the compiled graph."""
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id must be a non-empty string")

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Only the new user message is sent. The checkpointer restores the
        # earlier history, and the reducer appends this message to it.
        # iterations/answer are reset so limits and results are per request.
        result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=user_message.strip())],
                "iterations": 0,
                "answer": "",
            },
            config=config,
        )

        if result["answer"]:
            return result["answer"]

        final_message = result["messages"][-1]
        if isinstance(final_message.content, str):
            return final_message.content

        return str(final_message.content)