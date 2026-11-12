import pytest

from docqna.retrieval import RetrievalConfidenceGate, RetrievalResult


def test_confidence() -> None:
    def make_result(
        chunk_id: str, rank: int, score: float | None, chunk_index: int
    ) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            text=f"Text for {chunk_id}",
            rank=rank,
            distance=(-score if score is not None else 0.0),
            score=score,
            retrieval_method=("cross_encoder_rerank"),
            filename="sample.pdf",
            file_type="pdf",
            chunk_index=chunk_index,
        )

    gate = RetrievalConfidenceGate(min_top_score=1.5)

    # --------------------------------
    # 1. Strong retrieval
    # --------------------------------

    strong_results = [
        make_result(chunk_id="strong_1", rank=1, score=5.2, chunk_index=0),
        make_result(chunk_id="strong_2", rank=2, score=3.1, chunk_index=1),
    ]

    decision = gate.evaluate(strong_results)

    assert decision.is_confident

    assert decision.reason == ("score_threshold_passed")

    assert decision.top_score == 5.2

    assert decision.second_score == 3.1

    assert decision.score_margin is not None

    assert abs(decision.score_margin - 2.1) < 1e-9

    # --------------------------------
    # 2. Weak retrieval
    # --------------------------------

    weak_results = [
        make_result(chunk_id="weak_1", rank=1, score=-2.4, chunk_index=0),
        make_result(chunk_id="weak_2", rank=2, score=-4.0, chunk_index=1),
    ]

    decision = gate.evaluate(weak_results)

    assert not decision.is_confident

    assert decision.reason == ("below_score_threshold")

    # --------------------------------
    # 3. No results
    # --------------------------------

    decision = gate.evaluate([])

    assert not decision.is_confident

    assert decision.reason == ("no_results")

    # --------------------------------
    # 4. Missing score
    # --------------------------------

    missing_score_results = [
        make_result(chunk_id="missing_score", rank=1, score=None, chunk_index=0)
    ]

    decision = gate.evaluate(missing_score_results)

    assert not decision.is_confident

    assert decision.reason == ("missing_top_score")

    # --------------------------------
    # 5. Invalid configuration
    # --------------------------------

    with pytest.raises(ValueError):
        RetrievalConfidenceGate(min_top_score=float("nan"))
