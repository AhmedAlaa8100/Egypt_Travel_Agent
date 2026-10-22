
from openai import OpenAI

from travel_agent.config import get_settings


class LLMClient:
    """Client responsible for communicating with the configured LLM provider."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Add it to your .env file."
            )

        self._client = OpenAI(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
        )
        self._model = settings.groq_model

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ):
        """Generate a response, optionally providing tool definitions."""

        request = {
            "model": self._model,
            "messages": messages,
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        return self._client.chat.completions.create(**request)
