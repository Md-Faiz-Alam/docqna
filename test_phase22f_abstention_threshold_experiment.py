from docqna.retrieval.models import RetrievalResult

from run_phase22f_abstention_threshold_experiment import (
    ScoredCase,
    evaluate_threshold,
    score_summary,
    threshold_values,
)


def make_result(
    *,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        text="synthetic",
        rank=1,
        distance=-score,
        score=score,
        retrieval_method=(
            "cross_encoder_rerank"
        ),
        filename="sample.pdf",
        file_type="pdf",
        chunk_index=0,
        page_start=1,
        page_end=1,
    )


def make_case(
    *,
    case_id: str,
    should_answer: bool,
    score: float,
) -> ScoredCase:
    return ScoredCase(
        case_id=case_id,
        question="Synthetic question?",
        should_answer=should_answer,
        top_score=score,
        second_score=None,
        score_margin=None,
        result_count=1,
        retrieval_seconds=0.0,
        results=[
            make_result(
                score=score
            )
        ],
    )


def main() -> None:
    assert 0.0 in threshold_values()

    assert threshold_values() == sorted(
        threshold_values()
    )

    print(
        "Threshold matrix configuration: PASS"
    )

    cases = [
        make_case(
            case_id="answerable_high",
            should_answer=True,
            score=5.0,
        ),
        make_case(
            case_id="answerable_low",
            should_answer=True,
            score=1.0,
        ),
        make_case(
            case_id="unanswerable_low",
            should_answer=False,
            score=-5.0,
        ),
        make_case(
            case_id="unanswerable_high",
            should_answer=False,
            score=0.5,
        ),
    ]

    at_zero = evaluate_threshold(
        threshold=0.0,
        scored_cases=cases,
    )

    assert at_zero["true_answers"] == 2
    assert at_zero["true_abstentions"] == 1
    assert at_zero["false_answers"] == 1
    assert at_zero["false_abstentions"] == 0

    assert at_zero["accuracy"] == 0.75
    assert at_zero["answer_recall"] == 1.0

    print(
        "Confusion-matrix accounting: PASS"
    )

    at_two = evaluate_threshold(
        threshold=2.0,
        scored_cases=cases,
    )

    assert at_two["true_answers"] == 1
    assert at_two["true_abstentions"] == 2
    assert at_two["false_answers"] == 0
    assert at_two["false_abstentions"] == 1

    print(
        "Threshold trade-off behavior: PASS"
    )

    distribution = score_summary(
        cases
    )

    answerable = distribution[
        "answerable"
    ]

    unanswerable = distribution[
        "unanswerable"
    ]

    assert isinstance(
        answerable,
        dict,
    )
    assert isinstance(
        unanswerable,
        dict,
    )

    assert answerable["min"] == 1.0
    assert answerable["max"] == 5.0
    assert unanswerable["min"] == -5.0
    assert unanswerable["max"] == 0.5
    assert (
        distribution[
            "separation_gap"
        ]
        == 0.5
    )

    print(
        "Score-distribution summary: PASS"
    )

    print()
    print(
        "Phase 22F abstention-threshold tests passed."
    )


if __name__ == "__main__":
    main()
