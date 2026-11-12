import logging
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field, field_validator

from docqna.api.session import require_session_id
from docqna.feedback import FeedbackStore
from docqna.observability import FEEDBACK
from docqna.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=8000)
    abstained: bool = False
    cited_source_ids: list[int] = Field(default_factory=list, max_length=20)
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("comment")
    @classmethod
    def blank_comment_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class FeedbackResponse(BaseModel):
    feedback_id: int


@lru_cache(maxsize=1)
def get_feedback_store() -> FeedbackStore:
    settings = get_settings()
    return FeedbackStore(
        settings.metadata_dir / "feedback.db", retention_days=settings.feedback_retention_days
    )


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(
    request: FeedbackRequest,
    session_id: str = Depends(require_session_id),
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackResponse:
    """Record a thumbs up/down on an answer, to find and fix weak answers."""

    feedback_id = store.add(
        session_id=session_id,
        rating=request.rating,
        question=request.question,
        answer=request.answer,
        abstained=request.abstained,
        cited_source_ids=request.cited_source_ids,
        comment=request.comment,
    )
    FEEDBACK.labels(rating=request.rating).inc()
    logger.info("feedback_recorded rating=%s abstained=%s", request.rating, request.abstained)
    return FeedbackResponse(feedback_id=feedback_id)
