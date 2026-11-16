"""Signed assistant turns in conversation history.

Clients send earlier turns back with each follow-up question, and those turns
feed the question-rewriting prompt. User turns are the user's own words, but
an "assistant" turn should only be something this API actually answered:
otherwise a caller can invent assistant text and use it to steer the rewriter.

Every answer the API returns carries `answer_signature`, an HMAC of the
session ID and the answer text. Assistant turns whose signature does not
match are dropped before the rewriter sees them. Nothing is stored on the
server, so this works with a container that scales to zero. Without a
configured `HISTORY_SIGNING_KEY` the key is random per process, so after a
restart earlier answers simply stop counting as history.
"""

import hashlib
import hmac
import logging
import secrets
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from docqna.observability import UNSIGNED_HISTORY_TURNS
from docqna.settings import get_settings

logger = logging.getLogger(__name__)


class Turn(Protocol):
    @property
    def role(self) -> str: ...
    @property
    def content(self) -> str: ...
    @property
    def signature(self) -> str | None: ...


@lru_cache(maxsize=1)
def signing_key() -> bytes:
    configured = get_settings().history_signing_key
    if configured is not None and configured.get_secret_value():
        return configured.get_secret_value().encode("utf-8")
    return secrets.token_bytes(32)


def sign_answer(session_id: str, answer: str, key: bytes | None = None) -> str:
    message = f"{session_id}\x00{answer}".encode()
    return hmac.new(key or signing_key(), message, hashlib.sha256).hexdigest()


def verified_history(
    turns: Sequence[Turn], session_id: str, key: bytes | None = None
) -> list[tuple[str, str]]:
    """(role, text) turns to use: all user turns, and correctly signed assistant turns."""

    kept: list[tuple[str, str]] = []
    dropped = 0
    for turn in turns:
        if turn.role == "assistant":
            expected = sign_answer(session_id, turn.content, key)
            if not turn.signature or not hmac.compare_digest(turn.signature, expected):
                dropped += 1
                continue
        kept.append((turn.role, turn.content))
    if dropped:
        UNSIGNED_HISTORY_TURNS.inc(dropped)
        logger.warning("history_turns_dropped reason=bad_signature count=%d", dropped)
    return kept
