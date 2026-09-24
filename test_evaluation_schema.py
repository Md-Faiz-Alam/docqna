import json
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from docqna.evaluation import (
    EvaluationCase,
    load_evaluation_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parent

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)


def test_v2_dataset() -> None:
    """Verify the new 30-case Phase 21 dataset."""

    path = (
        EVALUATION_DIR
        / "evaluation_dataset_v2.json"
    )

    cases = load_evaluation_cases(
        path
    )

    assert len(cases) == 30

    answerable = [
        case
        for case in cases
        if case.should_answer
    ]

    unanswerable = [
        case
        for case in cases
        if not case.should_answer
    ]

    assert len(answerable) == 18
    assert len(unanswerable) == 12

    assert all(
        case.relevant_pages
        or case.relevant_chunk_ids
        for case in answerable
    )

    assert all(
        not case.relevant_pages
        and not case.relevant_chunk_ids
        for case in unanswerable
    )

    print(
        "Phase 21 v2 dataset: PASS"
    )


def test_legacy_answerable_compatibility() -> None:
    """Legacy baseline questions remain valid."""

    path = (
        EVALUATION_DIR
        / "baseline_questions.json"
    )

    cases = load_evaluation_cases(
        path
    )

    assert len(cases) == 18
    assert all(
        case.should_answer
        for case in cases
    )

    print(
        "Legacy answerable compatibility: PASS"
    )


def test_legacy_unanswerable_compatibility() -> None:
    """Legacy unanswerable questions are inferred correctly."""

    path = (
        EVALUATION_DIR
        / "unanswerable_questions.json"
    )

    cases = load_evaluation_cases(
        path
    )

    assert len(cases) == 12
    assert all(
        not case.should_answer
        for case in cases
    )

    print(
        "Legacy unanswerable compatibility: PASS"
    )


def test_invalid_answerable_case() -> None:
    """Answerable cases require retrieval ground truth."""

    try:
        EvaluationCase.model_validate(
            {
                "case_id": "invalid_answerable",
                "question": "Question?",
                "should_answer": True,
                "relevant_pages": [],
                "relevant_chunk_ids": [],
            }
        )

    except ValidationError:
        print(
            "Invalid answerable rejection: PASS"
        )

    else:
        raise AssertionError(
            "Answerable case without retrieval labels "
            "was not rejected."
        )


def test_invalid_unanswerable_case() -> None:
    """Unanswerable cases cannot claim relevant evidence."""

    try:
        EvaluationCase.model_validate(
            {
                "case_id": "invalid_unanswerable",
                "question": "Question?",
                "should_answer": False,
                "relevant_pages": [1],
            }
        )

    except ValidationError:
        print(
            "Invalid unanswerable rejection: PASS"
        )

    else:
        raise AssertionError(
            "Unanswerable case with relevant pages "
            "was not rejected."
        )


def test_duplicate_case_ids() -> None:
    """Dataset loader rejects duplicate case IDs."""

    payload = [
        {
            "case_id": "duplicate",
            "question": "First?",
            "should_answer": False,
        },
        {
            "case_id": "duplicate",
            "question": "Second?",
            "should_answer": False,
        },
    ]

    with TemporaryDirectory(
        ignore_cleanup_errors=True,
    ) as temp_dir:
        path = (
            Path(temp_dir)
            / "duplicate.json"
        )

        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        try:
            load_evaluation_cases(
                path
            )

        except ValueError:
            print(
                "Duplicate case ID rejection: PASS"
            )

        else:
            raise AssertionError(
                "Duplicate case IDs were not rejected."
            )


def main() -> None:
    test_v2_dataset()
    test_legacy_answerable_compatibility()
    test_legacy_unanswerable_compatibility()
    test_invalid_answerable_case()
    test_invalid_unanswerable_case()
    test_duplicate_case_ids()

    print()
    print(
        "Phase 21A evaluation schema tests passed."
    )


if __name__ == "__main__":
    main()
