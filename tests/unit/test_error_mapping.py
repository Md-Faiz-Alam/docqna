import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from docqna.api.errors import ERROR_SPECS
from docqna.api.main import create_app
from docqna.exceptions import DocQNAError


class UnmappedError(DocQNAError):
    """A domain error subclass with no explicit mapping."""


def _client_raising(exc: Exception) -> TestClient:
    app = create_app()
    router = APIRouter()

    @router.get("/boom")
    def boom() -> None:
        raise exc

    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize("exc_type", list(ERROR_SPECS), ids=lambda t: t.__name__)
def test_every_domain_error_has_stable_http_contract(exc_type: type[DocQNAError]) -> None:
    spec = ERROR_SPECS[exc_type]

    response = _client_raising(exc_type("details")).get("/boom")

    assert response.status_code == spec.status_code
    assert response.json() == {"error": spec.code, "detail": "details"}


def test_unmapped_subclass_falls_back_to_generic_error() -> None:
    response = _client_raising(UnmappedError("x")).get("/boom")

    assert response.status_code == 500
    assert response.json()["error"] == "docqna_error"
