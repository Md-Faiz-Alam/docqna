from typing import cast

from docqna.evaluation import EvaluationCase, SystemEvaluator
from docqna.exceptions import GenerationError
from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.pipeline import RAGPipeline
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class FakeRetriever:
    """Deterministic retriever for system-evaluation tests."""

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        if query == "abstain_case":
            return []

        return [
            RetrievalResult(
                chunk_id="chunk_1",
                text=("Synthetic evidence supporting the generated answer."),
                rank=1,
                distance=-5.0,
                score=5.0,
                retrieval_method="test",
                filename="sample.pdf",
                file_type="pdf",
                chunk_index=0,
                document_id="doc_test",
                document_version=1,
                page_start=1,
                page_end=1,
                section=None,
                source_segment_indices=[0],
                sections=[],
            )
        ]


class FakeLLM(BaseLLMProvider):
    """Deterministic generation provider."""

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.generate_calls += 1

        if "error_case" in prompt:
            raise GenerationError("Synthetic generation failure.")

        return "Synthetic supported answer [1]"

    def health_check(self) -> bool:
        return True


def test_system_evaluation() -> None:
    retriever = FakeRetriever()
    llm = FakeLLM()

    pipeline = RAGPipeline(retriever=cast(BaseRetriever, retriever), llm=llm, confidence_gate=None)

    evaluator = SystemEvaluator(rag_pipeline=pipeline)

    cases = [
        EvaluationCase(
            case_id="answered", question="answered_case", should_answer=True, relevant_pages=[1]
        ),
        EvaluationCase(case_id="abstained", question="abstain_case", should_answer=False),
        EvaluationCase(
            case_id="error", question="error_case", should_answer=True, relevant_pages=[1]
        ),
    ]

    report = evaluator.evaluate(cases=cases, top_k=5)

    assert report.num_cases == 3
    assert report.successful_cases == 2
    assert report.failed_cases == 1

    assert report.answered_cases == 1
    assert report.abstained_cases == 1

    assert abs(report.success_rate - (2 / 3)) < 1e-12

    assert abs(report.error_rate - (1 / 3)) < 1e-12

    assert report.abstention_rate == 0.5

    assert report.mean_retrieval_elapsed_seconds >= 0.0
    assert report.mean_generation_elapsed_seconds >= 0.0
    assert report.mean_postprocessing_elapsed_seconds >= 0.0
    assert report.mean_total_elapsed_seconds >= 0.0

    assert report.total_elapsed_seconds >= 0.0

    # Generation is not called for the abstention case.
    # It is called once for the successful answer and once
    # for the intentional generation-error case.
    assert llm.generate_calls == 2

    filtered_report = evaluator.evaluate(
        cases=[
            EvaluationCase(
                case_id="filtered", question="answered_case", should_answer=True, relevant_pages=[1]
            )
        ],
        top_k=5,
        metadata_filters={"document_id": "doc_test"},
    )

    assert filtered_report.successful_cases == 1
