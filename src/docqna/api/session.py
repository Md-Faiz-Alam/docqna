"""Anonymous session namespace enforced on every document/QA route.

This is isolation, not authentication: the header scopes documents and
retrieval to one high-entropy browser namespace.
"""

import re

from fastapi import Header, HTTPException, status

SESSION_HEADER_NAME = "X-DocQNA-Session-ID"
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def require_session_id(
    x_docqna_session_id: str | None = Header(default=None, alias=SESSION_HEADER_NAME),
) -> str:
    """Validate and return the anonymous session namespace."""

    if x_docqna_session_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required {SESSION_HEADER_NAME} header.",
        )

    session_id = x_docqna_session_id.strip()
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {SESSION_HEADER_NAME} header."
        )

    return session_id
