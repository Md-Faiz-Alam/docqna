from experiments.compare_grounding_checks import _clean, outcomes


def record(should_answer: bool, correct: bool, top: float | None, score: float | None) -> dict:
    return {
        "should_answer": should_answer,
        "correct": correct,
        "top_score": top,
        "whole_score": score,
        "sentence_score": score,
    }


def test_outcomes_apply_gate_and_grounding_threshold() -> None:
    records = [
        record(True, True, 0.0, 5.0),  # shown, correct
        record(True, False, 0.0, 5.0),  # shown, wrong
        record(True, True, 0.0, 1.0),  # withheld by grounding
        record(False, False, 0.0, 4.0),  # shown: made up
        record(False, False, -20.0, 9.0),  # stopped by the retrieval gate
        record(False, False, 0.0, None),  # refused by the LLM
    ]

    result = outcomes(records, "sentence_score", threshold=3.0, gate=-10.5)

    assert result == {
        "threshold": 3.0,
        "correct": 1,
        "wrong": 1,
        "made_up": 1,
        "answerable": 3,
        "unanswerable": 3,
    }


def test_clean_normalises_grouped_citations_like_the_pipeline() -> None:
    assert _clean("ANSWER: Two facts [1, 5].") == "Two facts [1][5]."
