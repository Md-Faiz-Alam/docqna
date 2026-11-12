import pytest

from docqna.evaluation.models import EvaluationCase
from docqna.filters import MetadataFilters
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from experiments.run_phase22e_topk_experiment import (
    TopKCaseMetrics,
    evaluate_case,
    summarize,
    top_k_values,
)


class FakeRetriever(BaseRetriever):
    """Deterministic ranked results for top-k metric tests."""

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        results = [
            RetrievalResult(
                chunk_id="chunk_1",
                text="irrelevant",
                rank=1,
                distance=0.1,
                filename="sample.pdf",
                file_type="pdf",
                chunk_index=0,
                page_start=1,
                page_end=1,
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                text="relevant",
                rank=2,
                distance=0.2,
                filename="sample.pdf",
                file_type="pdf",
                chunk_index=1,
                page_start=5,
                page_end=5,
            ),
            RetrievalResult(
                chunk_id="chunk_3",
                text="relevant",
                rank=3,
                distance=0.3,
                filename="sample.pdf",
                file_type="pdf",
                chunk_index=2,
                page_start=6,
                page_end=6,
            ),
        ]

        return results[:top_k]


def test_phase22e_topk_experiment() -> None:
    assert top_k_values() == [3, 5, 10, 20]

    case = EvaluationCase(
        case_id="case_001",
        question="Synthetic question?",
        should_answer=True,
        relevant_pages=[5, 6],
    )

    metrics = evaluate_case(retriever=FakeRetriever(), case=case, top_k=3)

    assert metrics.hit_at_k == 1.0
    assert metrics.reciprocal_rank_at_k == 0.5
    assert abs(metrics.precision_at_k - (2 / 3)) < 1e-12
    assert metrics.recall_at_k == 1.0
    assert metrics.page_recall_at_k == 1.0
    assert metrics.returned_results == 3

    summary = summarize(
        top_k=3,
        cases=[
            metrics,
            TopKCaseMetrics(
                case_id="case_002",
                top_k=3,
                hit_at_k=1.0,
                reciprocal_rank_at_k=1.0,
                precision_at_k=1.0 / 3.0,
                recall_at_k=0.5,
                ndcg_at_k=1.0,
                page_recall_at_k=0.5,
                returned_results=3,
                retrieval_seconds=0.1,
            ),
        ],
    )

    assert summary["evaluated_cases"] == 2
    assert summary["mean_hit_at_k"] == 1.0
    assert summary["mean_recall_at_k"] == 0.75
    assert summary["mean_page_recall_at_k"] == 0.75

    with pytest.raises(ValueError):
        summarize(top_k=5, cases=[])
