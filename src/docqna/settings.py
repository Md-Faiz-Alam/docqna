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
    max_upload_size_mb: int = 20
    max_question_chars: int = 2000
    max_documents_per_session: int = 3
    max_active_documents: int = 25
    max_chunks_per_document: int = 1000
    document_retention_hours: int = 24

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