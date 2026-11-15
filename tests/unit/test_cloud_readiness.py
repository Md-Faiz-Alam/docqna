"""Cloud deployment features: CORS, /info, sample documents, sample quota exemption."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import docqna.api.main as api_main
from docqna.api.dependencies import get_document_index_service_provider
from docqna.api.main import SESSION_HEADER_NAME, app
from docqna.api.samples import sample_content_hashes, sample_document_paths
from docqna.embeddings import BaseEmbeddingModel
from docqna.exceptions import DuplicateDocumentError, ResourceLimitError
from docqna.indexing import ChromaVectorStore, DocumentIndexingResult, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRecord, DocumentRegistry
from docqna.settings import Settings

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data/evaluation/public/corpus"
SESSION = "cloud-session-" + "b" * 32
WEBSITE = "https://docqna.example.app"


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"sample_documents_dir": SAMPLES_DIR, **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


def make_record(name: str, session_id: str, content_hash: str) -> DocumentRecord:
    return DocumentRecord(
        document_id=f"doc_{name}",
        session_id=session_id,
        filename=name,
        file_type=Path(name).suffix.lstrip("."),
        version=1,
        content_hash=content_hash,
        size_bytes=1,
        registered_at=datetime.now(UTC),
    )


# --------------------------------------------------------------------- CORS


@pytest.fixture
def cors_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    settings = make_settings(
        cors_allowed_origins=f"{WEBSITE}/, http://localhost:3000",
        cors_allowed_origin_regex=r"https://docqna-[a-z0-9-]+\.vercel\.app",
        rate_limit_qa_per_minute=1,
    )
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    return api_main.create_app()


def preflight(client: TestClient, origin: str) -> dict[str, str]:
    response = client.options(
        "/qa",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{SESSION_HEADER_NAME}",
        },
    )
    return dict(response.headers)


def test_cors_allows_only_configured_origins(cors_app: FastAPI) -> None:
    client = TestClient(cors_app)

    allowed = preflight(client, WEBSITE)
    assert allowed["access-control-allow-origin"] == WEBSITE
    assert SESSION_HEADER_NAME.lower() in allowed["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in allowed

    preview = preflight(client, "https://docqna-git-feature-x.vercel.app")
    assert preview["access-control-allow-origin"] == "https://docqna-git-feature-x.vercel.app"

    assert "access-control-allow-origin" not in preflight(client, "https://evil.example.com")


def test_cors_headers_on_rate_limited_responses(cors_app: FastAPI) -> None:
    client = TestClient(cors_app)
    headers = {"Origin": WEBSITE, SESSION_HEADER_NAME: SESSION}

    client.post("/qa", headers=headers, json={"question": ""})
    limited = client.post("/qa", headers=headers, json={"question": "hello"})

    assert limited.status_code == 429
    assert limited.headers["access-control-allow-origin"] == WEBSITE


def test_default_settings_allow_no_cross_origin_calls() -> None:
    assert Settings(_env_file=None).cors_origins == []  # type: ignore[call-arg]


# --------------------------------------------------------------------- /info


def test_info_describes_deployment_without_secrets(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    settings = make_settings(llm_provider="gemini", gemini_api_key="super-secret-key")
    monkeypatch.setattr("docqna.api.routes.system.get_settings", lambda: settings)

    response = client.get("/info")

    assert response.status_code == 200
    body = response.json()
    assert body["llm_provider"] == "gemini"
    assert body["llm_model"] == settings.gemini_model
    assert body["external_llm"] is True
    assert body["sample_documents"] == [p.name for p in sample_document_paths(SAMPLES_DIR)]
    assert body["limits"]["supported_file_types"] == ["docx", "pdf", "txt"]
    assert body["limits"]["max_documents_per_session"] == settings.max_documents_per_session
    assert "super-secret-key" not in response.text


def test_info_for_local_ollama(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setattr("docqna.api.routes.system.get_settings", make_settings)

    body = client.get("/info").json()

    assert body["llm_provider"] == "ollama"
    assert body["external_llm"] is False


# ----------------------------------------------------------- sample documents


class RecordingIndexService:
    def __init__(self) -> None:
        self.records: dict[tuple[str | None, str], DocumentRecord] = {}

    def index_document(
        self, path: str | Path, session_id: str | None = None
    ) -> DocumentIndexingResult:
        name = Path(path).name
        if (session_id, name) in self.records:
            raise DuplicateDocumentError("already registered")
        record = make_record(name, session_id or "", "c" * 64)
        self.records[(session_id, name)] = record
        return DocumentIndexingResult(record=record, chunk_count=1)

    def list_documents(self, session_id: str | None = None) -> list[DocumentRecord]:
        return [r for (s, _), r in self.records.items() if s == session_id]

    def purge_expired_documents(self) -> int:
        return 0


@pytest.fixture
def samples_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, RecordingIndexService]]:
    service = RecordingIndexService()
    monkeypatch.setattr("docqna.api.routes.documents.get_settings", make_settings)
    monkeypatch.setattr("docqna.api.routes.documents.invalidate_session_index", lambda _: None)
    app.dependency_overrides[get_document_index_service_provider] = lambda: lambda: service
    app.state.rate_limiter.reset()
    try:
        yield TestClient(app, headers={SESSION_HEADER_NAME: SESSION}), service
    finally:
        app.dependency_overrides.clear()


def test_load_samples_indexes_each_bundled_document_once(
    samples_client: tuple[TestClient, RecordingIndexService],
) -> None:
    client, service = samples_client
    expected = sorted(p.name for p in sample_document_paths(SAMPLES_DIR))
    assert len(expected) == 4

    first = client.post("/documents/samples")
    second = client.post("/documents/samples")

    assert first.status_code == second.status_code == 200
    assert sorted(d["filename"] for d in first.json()["documents"]) == expected
    assert second.json()["count"] == 4
    assert len(service.records) == 4


def test_load_samples_404_when_none_are_bundled(
    monkeypatch: pytest.MonkeyPatch,
    samples_client: tuple[TestClient, RecordingIndexService],
    tmp_path: Path,
) -> None:
    client, _ = samples_client
    monkeypatch.setattr(
        "docqna.api.routes.documents.get_settings",
        lambda: make_settings(sample_documents_dir=tmp_path),
    )

    assert client.post("/documents/samples").status_code == 404


# ---------------------------------------------------- quota exemption


class HashingRegistry:
    compute_content_hash = staticmethod(DocumentRegistry.compute_content_hash)

    def __init__(self, records: list[DocumentRecord]) -> None:
        self.records = records

    def list_latest_records(self, session_id: str | None = None) -> list[DocumentRecord]:
        return [r for r in self.records if session_id is None or r.session_id == session_id]


def quota_service(records: list[DocumentRecord], *, active: int = 25) -> DocumentIndexService:
    return DocumentIndexService(
        ingestion_service=cast(DocumentIngestionService, object()),
        registry=cast(DocumentRegistry, HashingRegistry(records)),
        embedding_model=cast(BaseEmbeddingModel, object()),
        vector_store=cast(ChromaVectorStore, object()),
        max_documents_per_session=3,
        max_active_documents=active,
        limit_exempt_hashes=sample_content_hashes(SAMPLES_DIR),
    )


def test_samples_do_not_use_or_need_quota(tmp_path: Path) -> None:
    sample_hashes = sorted(sample_content_hashes(SAMPLES_DIR))
    own = tmp_path / "own.txt"
    own.write_text("my own document", encoding="utf-8")

    # A session holding 4 samples can still upload its own documents.
    samples = [make_record(f"s{i}.txt", SESSION, h) for i, h in enumerate(sample_hashes)]
    quota_service(samples)._enforce_document_limits(own, SESSION)  # pyright: ignore[reportPrivateUsage]

    # A session at its own-document limit can still load samples...
    full = [make_record(f"own{i}.txt", SESSION, str(i) * 64) for i in range(3)]
    service = quota_service(full, active=3)
    service._enforce_document_limits(sample_document_paths(SAMPLES_DIR)[0], SESSION)  # pyright: ignore[reportPrivateUsage]

    # ...but not another document of its own.
    with pytest.raises(ResourceLimitError):
        service._enforce_document_limits(own, SESSION)  # pyright: ignore[reportPrivateUsage]


def test_root_points_visitors_to_the_docs(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
