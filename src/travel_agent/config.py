
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_persist_directory: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma"
    )
    memory_collection: str = "my_full_agent_memory"
    document_collection: str = "my_agent_docs"

    agent_max_iterations: int = Field(default=5, ge=1, le=50)
    tool_max_retries: int = Field(default=3, ge=1, le=10)


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance per process."""
    return Settings()
