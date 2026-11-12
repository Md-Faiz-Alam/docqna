import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError

from docqna.evaluation import EvaluationCase, load_evaluation_cases

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"


def test_v2_dataset() -> None:
    """Verify the new 30-case Phase 21 dataset."""

    path = EVALUATION_DIR / "evaluation_dataset_v2.json"

    cases = load_evaluation_cases(path)

    assert len(cases) == 30

    answerable = [case for case in cases if case.should_answer]

    unanswerable = [case for case in cases if not case.should_answer]

    assert len(answerable) == 18
    assert len(unanswerable) == 12

    assert all(case.relevant_pages or case.relevant_chunk_ids for case in answerable)

    assert all(not case.relevant_pages and not case.relevant_chunk_ids for case in unanswerable)


def test_invalid_answerable_case() -> None:
    """Answerable cases require retrieval ground truth."""

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(
            {
                "case_id": "invalid_answerable",
                "question": "Question?",
                "should_answer": True,
                "relevant_pages": [],
                "relevant_chunk_ids": [],
            }
        )


def test_invalid_unanswerable_case() -> None:
    """Unanswerable cases cannot claim relevant evidence."""

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(
            {
                "case_id": "invalid_unanswerable",
                "question": "Question?",
                "should_answer": False,
                "relevant_pages": [1],
            }
        )


def test_duplicate_case_ids() -> None:
    """Dataset loader rejects duplicate case IDs."""

    payload = [
        {"case_id": "duplicate", "question": "First?", "should_answer": False},
        {"case_id": "duplicate", "question": "Second?", "should_answer": False},
    ]

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        path = Path(temp_dir) / "duplicate.json"

        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError):
            load_evaluation_cases(path)
