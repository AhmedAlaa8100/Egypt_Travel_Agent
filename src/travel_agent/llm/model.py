from langchain_openai import ChatOpenAI

from travel_agent.config import get_settings


def create_chat_model() -> ChatOpenAI:
    """Create a chat model configured with the project's settings."""
    settings = get_settings()

    if not settings.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to the project-root .env file."
        )

    return ChatOpenAI(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=settings.groq_base_url,
        temperature=0,
    )
