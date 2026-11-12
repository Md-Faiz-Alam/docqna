"""Shared pytest fixtures for the DocQ&A test suite."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from docqna.api.dependencies import get_document_index_service_provider
from docqna.api.main import SESSION_HEADER_NAME, app

TEST_SESSION_ID = "test-session-" + "a" * 32


class NullIndexService:
    """Index service stand-in for routes that only need retention purging."""

    def purge_expired_documents(self) -> int:
        return 0

    def list_documents(self, session_id: str | None = None) -> list:
        return []

    def delete_document(self, document_id: str, session_id: str | None = None) -> bool:
        return False


@pytest.fixture
def session_id() -> str:
    return TEST_SESSION_ID


@pytest.fixture
def client(session_id: str) -> Iterator[TestClient]:
    """API client with a valid session header and no model loading."""

    app.dependency_overrides[get_document_index_service_provider] = lambda: NullIndexService
    app.state.rate_limiter.reset()
    try:
        with TestClient(app, headers={SESSION_HEADER_NAME: session_id}) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
