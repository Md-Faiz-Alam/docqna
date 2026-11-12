from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docqna.api.main import app
from docqna.api.routes.feedback import get_feedback_store
from docqna.feedback import FeedbackStore


@pytest.fixture
def store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback.db", retention_days=30)


def add(store: FeedbackStore, rating: str = "down", **overrides) -> int:
    fields = {
        "session_id": "s" * 32,
        "rating": rating,
        "question": "What is the meal cap?",
        "answer": "60 euros [1].",
        "abstained": False,
        "cited_source_ids": [1],
    }
    fields.update(overrides)
    return store.add(**fields)


def test_store_round_trip_and_filter(store: FeedbackStore) -> None:
    add(store, "up")
    add(store, "down", comment="Wrong page cited")

    downs = store.list(rating="down")

    assert len(store.list()) == 2
    assert [r.comment for r in downs] == ["Wrong page cited"]
    assert downs[0].cited_source_ids == [1]


def test_old_feedback_expires(store: FeedbackStore) -> None:
    now = datetime.now(UTC)
    add(store, now=now - timedelta(days=40))
    add(store, now=now)

    assert len(store.list()) == 1


def test_feedback_endpoint_records_rating(client: TestClient, store: FeedbackStore) -> None:
    app.dependency_overrides[get_feedback_store] = lambda: store

    response = client.post(
        "/feedback",
        json={
            "rating": "down",
            "question": "What is the meal cap?",
            "answer": "It is 50 euros [1].",
            "cited_source_ids": [1],
            "comment": "  Should be 60  ",
        },
    )

    assert response.status_code == 201
    assert response.json()["feedback_id"] >= 1
    assert store.list()[0].comment == "Should be 60"


@pytest.mark.parametrize(
    "body",
    [
        {"rating": "meh", "question": "q", "answer": "a"},
        {"rating": "up", "question": "", "answer": "a"},
        {"rating": "up", "question": "q", "answer": "a" * 9000},
    ],
)
def test_feedback_endpoint_validates_input(client: TestClient, body: dict) -> None:
    assert client.post("/feedback", json=body).status_code == 422
