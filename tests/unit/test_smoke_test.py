"""The deploy smoke test, against a fake API (the real one runs in the deploy pipeline)."""

from typing import Any

import pytest
from scripts import smoke_test
from scripts.smoke_test import SmokeTestFailure

ANSWER = {
    "answer": "The standard working week is 38 hours [1].",
    "abstained": False,
    "sources": [{"citation_id": 1}],
    "answer_signature": "ab" * 32,
}


class FakeApi:
    """Answers like the API; `limit_at` is the request number that gets 429."""

    def __init__(self, *, limit_at: int | None = 61, answer: dict | None = None) -> None:
        self.limit_at = limit_at
        self.answer = answer or ANSWER
        self.rate_checks = 0
        self.forwarded_for: set[str] = set()
        self.sessions: set[str] = set()

    def __call__(self, method: str, url: str, *, body: Any = None, headers=None, timeout=0):
        headers = headers or {}
        if url.endswith("/documents/samples"):
            return 200, {}, {"count": 4, "documents": []}
        if url.endswith("/qa") and body["question"] == "rate limit check":
            self.rate_checks += 1
            self.forwarded_for.add(headers["X-Forwarded-For"])
            self.sessions.add(headers[smoke_test.SESSION_HEADER])
            if self.limit_at is not None and self.rate_checks >= self.limit_at:
                return 429, {}, {"error": "rate_limited"}
            return 200, {}, {"abstained": True}
        if url.endswith("/qa"):
            return 200, {}, self.answer
        raise AssertionError(url)


def test_rate_limit_check_passes_when_a_429_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeApi(limit_at=61)
    monkeypatch.setattr(smoke_test, "call", api)

    smoke_test.check_ip_rate_limit("https://x")

    assert api.rate_checks == 61
    # Every request looked like a new visitor with a new (fake) IP.
    assert len(api.sessions) == 61
    assert len(api.forwarded_for) == 61


def test_rate_limit_check_fails_when_the_limit_can_be_dodged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "call", FakeApi(limit_at=None))

    with pytest.raises(SmokeTestFailure, match="TRUSTED_PROXY_HOPS"):
        smoke_test.check_ip_rate_limit("https://x")


def test_real_answer_check_passes_for_a_cited_signed_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "call", FakeApi())

    smoke_test.check_real_answer("https://x")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"abstained": True}, "refused"),
        ({"sources": []}, "no sources"),
        ({"answer_signature": ""}, "not signed"),
    ],
)
def test_real_answer_check_fails_on_a_bad_answer(
    monkeypatch: pytest.MonkeyPatch, change: dict, message: str
) -> None:
    monkeypatch.setattr(smoke_test, "call", FakeApi(answer={**ANSWER, **change}))

    with pytest.raises(SmokeTestFailure, match=message):
        smoke_test.check_real_answer("https://x")


def test_wait_for_build_waits_until_the_new_commit_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    builds = iter(["old", "old", "new"])
    seen: list[str] = []

    def fake_call(method, url, **kwargs):
        seen.append(url)
        return 200, {}, {"build": next(builds)}

    monkeypatch.setattr(smoke_test, "call", fake_call)
    monkeypatch.setattr(smoke_test.time, "sleep", lambda _: None)

    smoke_test.wait_for_build("https://x", "new", wait_seconds=60)

    assert seen == ["https://x/info"] * 3


def test_wait_for_build_fails_when_the_old_version_keeps_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(smoke_test, "call", lambda *a, **k: (200, {}, {"build": "old"}))
    monkeypatch.setattr(smoke_test.time, "sleep", lambda _: None)
    monkeypatch.setattr(smoke_test.time, "monotonic", lambda: next(clock))

    with pytest.raises(SmokeTestFailure, match="still reports build 'old'"):
        smoke_test.wait_for_build("https://x", "new", wait_seconds=30)
