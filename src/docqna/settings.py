from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from docqna.constants import PROJECT_ROOT


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`.

    RAG defaults are the configuration selected by the phase-22
    evaluation (see docs/evaluation.md). Change them only with new
    evaluation evidence.
    """

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Storage. All runtime state lives under data_dir unless overridden.
    data_dir: Path = PROJECT_ROOT / "data"
    log_dir: Path = PROJECT_ROOT / "logs"
    chroma_collection: str = "docqna_documents"

    # Public deployment resource protection
    max_upload_size_mb: int = 20
    max_question_chars: int = 2000
    max_documents_per_session: int = 3
    max_active_documents: int = 25
    max_chunks_per_document: int = 1000
    document_retention_hours: int = 24
    # Per session (or client IP), per process. 0 disables a limit.
    rate_limit_qa_per_minute: int = 20
    rate_limit_uploads_per_minute: int = 10

    # LLM
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 320
    llm_timeout_seconds: float = 180.0

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # Retrieval (evaluation-selected production configuration)
    rrf_k: int = 60
    hybrid_candidate_pool_size: int = 10
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"
    reranker_candidate_pool_size: int = 20
    answerability_min_score: float = 0.0
    max_cached_session_indexes: int = 256

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_max_bytes: int = 5_000_000
    log_backup_count: int = 3

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "indexes" / "chroma"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def registry_path(self) -> Path:
        return self.metadata_dir / "documents.db"


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
