import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from docqna.evaluation import RetrievalCaseResult, RetrievalEvaluationReport, save_report_bundle


def build_report() -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport(
        num_cases=1,
        top_k=5,
        num_skipped_unanswerable=0,
        mean_hit_at_1=1.0,
        mean_hit_at_3=1.0,
        mean_hit_at_5=1.0,
        mean_reciprocal_rank=1.0,
        mean_page_recall_at_5=1.0,
        mean_precision_at_k=0.4,
        mean_recall_at_k=1.0,
        mean_ndcg_at_k=1.0,
        cases=[
            RetrievalCaseResult(
                case_id="case_001",
                question="Synthetic question?",
                hit_at_1=1.0,
                hit_at_3=1.0,
                hit_at_5=1.0,
                reciprocal_rank=1.0,
                page_recall_at_5=1.0,
                precision_at_k=0.4,
                recall_at_k=1.0,
                ndcg_at_k=1.0,
                retrieved_chunk_ids=["chunk_1", "chunk_2"],
                retrieved_pages=["1", "2"],
                distances=[-5.0, -4.0],
            )
        ],
    )


def test_evaluation_reporting() -> None:
    report = build_report()

    metadata = {
        "experiment": "phase21f_test",
        "retriever": "hybrid_reranker",
        "chunking": "fixed_200_40",
    }

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        output_dir = Path(temp_dir)

        paths = save_report_bundle(report, output_dir, stem="retrieval_report", metadata=metadata)

        assert set(paths) == {"json", "summary_csv", "cases_csv", "summary_text"}

        assert all(path.exists() for path in paths.values())

        payload = json.loads(paths["json"].read_text(encoding="utf-8"))

        assert payload["report_type"] == "RetrievalEvaluationReport"
        assert payload["metadata"]["experiment"] == "phase21f_test"
        assert payload["report"]["num_cases"] == 1
        assert payload["report"]["cases"][0]["case_id"] == "case_001"

        with paths["summary_csv"].open("r", encoding="utf-8", newline="") as file:
            summary_rows = list(csv.DictReader(file))

        summary = {row["metric"]: row["value"] for row in summary_rows}

        assert summary["report_type"] == "RetrievalEvaluationReport"
        assert summary["num_cases"] == "1"
        assert summary["mean_precision_at_k"] == "0.4"

        with paths["cases_csv"].open("r", encoding="utf-8", newline="") as file:
            case_rows = list(csv.DictReader(file))

        assert len(case_rows) == 1

        assert case_rows[0]["case_id"] == "case_001"

        assert case_rows[0]["retrieved_chunk_ids"] == '["chunk_1","chunk_2"]'

        summary_text = paths["summary_text"].read_text(encoding="utf-8")

        assert "report_type: RetrievalEvaluationReport" in summary_text
        assert "mean_ndcg_at_k: 1.0" in summary_text

        second_paths = save_report_bundle(
            report, output_dir, stem="retrieval_report_copy", metadata=metadata
        )

        first_payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        second_payload = json.loads(second_paths["json"].read_text(encoding="utf-8"))

        assert first_payload == second_payload

    with pytest.raises(ValueError):
        save_report_bundle(report, Path("."), stem="   ")
