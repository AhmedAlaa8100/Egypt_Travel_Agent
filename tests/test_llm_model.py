
from pydantic import SecretStr

from travel_agent.config import Settings
from travel_agent.llm.model import create_chat_model


def test_create_chat_model_uses_configured_settings(monkeypatch):
    settings = Settings(
        groq_api_key=SecretStr("test-key"),
        groq_base_url="https://example.com/v1",
        groq_model="test-model",
    )

    monkeypatch.setattr(
        "travel_agent.llm.model.get_settings",
        lambda: settings,
    )

    model = create_chat_model()

    assert model.model_name == "test-model"
    assert model.openai_api_base == "https://example.com/v1"
    assert model.temperature == 0


def test_create_chat_model_rejects_missing_api_key(monkeypatch):
    settings = Settings(groq_api_key=None)

    monkeypatch.setattr(
        "travel_agent.llm.model.get_settings",
        lambda: settings,
    )

    try:
        create_chat_model()
    except ValueError as exc:
        assert "GROQ_API_KEY is missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a missing API key")
