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
    # Commit the running image was built from (set by docker/cloud.Dockerfile).
    # Shown in /info so a deploy can confirm which version answers.
    build_sha: str = "dev"

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
    # Each client IP may make this many times the per-session limit, so a script
    # cannot dodge the limits by sending a new session ID each time. 0 disables it.
    rate_limit_ip_factor: int = 3
    # Proxies in front of the API that append to X-Forwarded-For (Azure Container
    # Apps ingress: 1). 0 uses the socket address and ignores the header.
    trusted_proxy_hops: int = 0

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
    # Score each answer sentence against the sources it cites, and use the weakest
    # (on) instead of scoring the whole answer as one claim (off). Same thresholds;
    # measured in experiments/compare_grounding_checks.py.
    grounding_per_sentence: bool = True
    # Prompt-injection defences (src/docqna/rag/injection.py), measured in
    # experiments/attack_prompt_injection.py.
    injection_filter_enabled: bool = True
    # Stream unverified draft tokens on /qa/stream before the final checks run.
    # Off by default: a draft can contain a made-up answer or an injected link
    # that the checks then withdraw, but the user has already seen it.
    stream_drafts: bool = False
    # Put the answer rules in the LLM's system role and wrap each document in a
    # tag with a random per-request name. Off until the attack and answerability
    # tests are re-run with it (it changes what the LLM sees).
    separate_prompt_roles: bool = False
    max_cached_session_indexes: int = 256
    # Key that signs answers so clients cannot forge assistant turns in history.
    # Unset: a random key per process (answers from before a restart stop counting).
    history_signing_key: SecretStr | None = None

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
