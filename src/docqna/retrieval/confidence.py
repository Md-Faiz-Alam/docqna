import math

from pydantic import BaseModel

from docqna.retrieval.models import RetrievalResult


class RetrievalConfidenceDecision(BaseModel):
    """
    Result of evaluating retrieval confidence.

    The current baseline confidence signal is the
    cross-encoder score of the rank-1 result.
    """

    is_confident: bool

    top_score: float | None = None
    second_score: float | None = None
    score_margin: float | None = None

    threshold: float

    reason: str


class RetrievalConfidenceGate:
    """
    Decide whether retrieved evidence is strong enough
    for the RAG system to attempt an answer.

    Phase 15 initially uses the rank-1 cross-encoder
    score as the confidence signal.

    The threshold remains configurable so it can be
    calibrated from evaluation data instead of guessed.
    """

    def __init__(
        self,
        min_top_score: float,
    ) -> None:
        if not math.isfinite(
            min_top_score
        ):
            raise ValueError(
                "min_top_score must be a finite number."
            )

        self.min_top_score = (
            min_top_score
        )

    @staticmethod
    def _get_score(
        result: RetrievalResult,
    ) -> float | None:
        """Return a valid finite retrieval score."""

        score = result.score

        if score is None:
            return None

        score = float(
            score
        )

        if not math.isfinite(
            score
        ):
            return None

        return score

    def evaluate(
        self,
        results: list[RetrievalResult],
    ) -> RetrievalConfidenceDecision:
        """
        Evaluate whether retrieval confidence is high
        enough to allow answer generation.
        """

        if not results:
            return RetrievalConfidenceDecision(
                is_confident=False,
                top_score=None,
                second_score=None,
                score_margin=None,
                threshold=self.min_top_score,
                reason="no_results",
            )

        ranked_results = sorted(
            results,
            key=lambda result: (
                result.rank,
                result.chunk_index,
                result.chunk_id,
            ),
        )

        top_score = self._get_score(
            ranked_results[0]
        )

        if top_score is None:
            return RetrievalConfidenceDecision(
                is_confident=False,
                top_score=None,
                second_score=None,
                score_margin=None,
                threshold=self.min_top_score,
                reason="missing_top_score",
            )

        second_score: float | None = None

        if len(ranked_results) >= 2:
            second_score = self._get_score(
                ranked_results[1]
            )

        score_margin: float | None = None

        if second_score is not None:
            score_margin = (
                top_score
                - second_score
            )

        if top_score < self.min_top_score:
            return RetrievalConfidenceDecision(
                is_confident=False,
                top_score=top_score,
                second_score=second_score,
                score_margin=score_margin,
                threshold=self.min_top_score,
                reason="below_score_threshold",
            )

        return RetrievalConfidenceDecision(
            is_confident=True,
            top_score=top_score,
            second_score=second_score,
            score_margin=score_margin,
            threshold=self.min_top_score,
            reason="score_threshold_passed",
        )