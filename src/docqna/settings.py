from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    log_level: str = "INFO"
    log_to_file: bool = True
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()