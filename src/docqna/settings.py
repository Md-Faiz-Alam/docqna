from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from docqna.constants import PROJECT_ROOT


class Settings(BaseSettings):
    """Application settings loaded from environment variables / `.env`.

    RAG defaults are the configuration selected by the retrieval
    experiments (see docs/evaluation.md). Change them only with new
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
    # Fictional demo documents a visitor can load with one click.
    sample_documents_dir: Path = PROJECT_ROOT / "data" / "evaluation" / "public" / "corpus"

    # Browser origins allowed to call the API (CORS), comma-separated, e.g.
    # "https://docqna.vercel.app". Empty allows no cross-origin browser calls.
    cors_allowed_origins: str = ""
    # Optional regex for extra origins, e.g. Vercel preview deployments.
    cors_allowed_origin_regex: str = ""

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
    rate_limit_feedback_per_minute: int = 30

    # Answer feedback (question/answer text is kept to build evaluation cases)
    feedback_retention_days: int = 30

    # OCR for scanned PDF pages (needs the Tesseract binary; installed in Docker)
    ocr_enabled: bool = True
    ocr_languages: str = "eng"
    max_ocr_pages: int = 50

    # LLM: "ollama" (local, private, default) or "gemini" (hosted, fast)
    llm_provider: Literal["ollama", "gemini"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    # "minimal" keeps answers fast and leaves the token budget for the answer;
    # empty uses the model's own default.
    gemini_thinking_level: str = "minimal"
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
    # Answerability calibration (end-to-end, LLM in the loop): a loose retrieval gate that only
    # skips clearly off-topic questions, plus a post-generation grounding check.
    answerability_min_score: float = -10.5
    grounding_check_enabled: bool = True
    # The grounding threshold depends on how the model phrases answers, so each
    # provider has its own measured value (Ollama phi3:mini / Gemini).
    grounding_min_score: float = 3.3
    gemini_grounding_min_score: float = 1.5
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
    def llm_model(self) -> str:
        """Model name of the active LLM provider."""
        return self.gemini_model if self.llm_provider == "gemini" else self.ollama_model

    @property
    def cors_origins(self) -> list[str]:
        """Allowed CORS origins, without trailing slashes."""
        return [
            origin.strip().rstrip("/")
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def active_grounding_min_score(self) -> float:
        """Grounding threshold measured for the active LLM provider."""
        if self.llm_provider == "gemini":
            return self.gemini_grounding_min_score
        return self.grounding_min_score

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
