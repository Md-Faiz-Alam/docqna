import pytest

from experiments.run_phase23_answerability_recalibration import outcomes_at

RECORDS = [
    # answerable: high score, correct
    {"should_answer": True, "top_score": 3.0, "answered": True, "cited_relevant": True},
    # answerable: low score, LLM would have answered correctly
    {"should_answer": True, "top_score": -4.0, "answered": True, "cited_relevant": True},
    # answerable: LLM answers with the wrong evidence
    {"should_answer": True, "top_score": 1.0, "answered": True, "cited_relevant": False},
    # unanswerable: LLM correctly abstains even past the gate
    {"should_answer": False, "top_score": 2.0, "answered": False, "cited_relevant": False},
    # unanswerable: LLM hallucinates if allowed through
    {"should_answer": False, "top_score": -3.0, "answered": True, "cited_relevant": False},
]


def test_strict_gate_blocks_low_scores() -> None:
    result = outcomes_at(RECORDS, 0.0)

    assert result["answerable_correct"] == pytest.approx(1 / 3)
    assert result["answerable_wrong_answer"] == pytest.approx(1 / 3)
    assert result["unanswerable_false_answer"] == 0.0


def test_loose_gate_trades_recall_for_false_answers() -> None:
    result = outcomes_at(RECORDS, -5.0)

    assert result["answerable_correct"] == pytest.approx(2 / 3)
    assert result["unanswerable_false_answer"] == pytest.approx(0.5)


def test_missing_score_always_abstains() -> None:
    records = [{"should_answer": True, "top_score": None, "answered": True, "cited_relevant": True}]

    assert outcomes_at(records, float("-inf"))["answerable_correct"] == 0.0
