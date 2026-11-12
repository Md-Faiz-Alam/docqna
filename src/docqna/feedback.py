"""User feedback on answers (thumbs up/down), stored in SQLite.

Feedback is the cheapest source of new evaluation cases: down-voted answers
show where retrieval or generation fails on real questions. Records expire
after `feedback_retention_days`.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from docqna.exceptions import DocQNAError

Rating = Literal["up", "down"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    session_id       TEXT    NOT NULL,
    rating           TEXT    NOT NULL CHECK (rating IN ('up', 'down')),
    question         TEXT    NOT NULL,
    answer           TEXT    NOT NULL,
    abstained        INTEGER NOT NULL,
    cited_source_ids TEXT    NOT NULL,
    comment          TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at);
"""


class FeedbackError(DocQNAError):
    """Feedback could not be stored or read."""


class FeedbackRecord(BaseModel):
    id: int
    created_at: datetime
    session_id: str
    rating: Rating
    question: str
    answer: str
    abstained: bool
    cited_source_ids: list[int]
    comment: str | None = None


class FeedbackStore:
    def __init__(self, db_path: str | Path, retention_days: int = 30) -> None:
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            raw = sqlite3.connect(self.db_path, timeout=30.0)
        except sqlite3.Error as exc:
            raise FeedbackError("Could not open feedback store.") from exc
        with closing(raw) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            try:
                with connection:  # commits on success, rolls back on error
                    yield connection
            except sqlite3.Error as exc:
                raise FeedbackError("Feedback store operation failed.") from exc

    def add(
        self,
        *,
        session_id: str,
        rating: Rating,
        question: str,
        answer: str,
        abstained: bool,
        cited_source_ids: list[int],
        comment: str | None = None,
        now: datetime | None = None,
    ) -> int:
        created_at = now or datetime.now(UTC)
        with self._connect() as connection:
            self._purge(connection, created_at)
            cursor = connection.execute(
                "INSERT INTO feedback (created_at, session_id, rating, question, answer, "
                "abstained, cited_source_ids, comment) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    created_at.isoformat(),
                    session_id,
                    rating,
                    question,
                    answer,
                    int(abstained),
                    json.dumps(cited_source_ids),
                    comment,
                ),
            )
            return int(cursor.lastrowid or 0)

    def _purge(self, connection: sqlite3.Connection, now: datetime) -> None:
        if self.retention_days > 0:
            cutoff = (now - timedelta(days=self.retention_days)).isoformat()
            connection.execute("DELETE FROM feedback WHERE created_at < ?", (cutoff,))

    def list(self, rating: Rating | None = None) -> list[FeedbackRecord]:
        query = "SELECT * FROM feedback"
        params: tuple[str, ...] = ()
        if rating is not None:
            query += " WHERE rating = ?"
            params = (rating,)
        with self._connect() as connection:
            rows = connection.execute(query + " ORDER BY id", params).fetchall()
        return [
            FeedbackRecord.model_validate(
                {
                    **dict(row),
                    "abstained": bool(row["abstained"]),
                    "cited_source_ids": json.loads(row["cited_source_ids"]),
                }
            )
            for row in rows
        ]
