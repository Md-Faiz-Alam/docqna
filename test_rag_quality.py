from docqna.evaluation import (
    EvaluationCase,
    RAGQualityEvaluator,
    RAGQualityJudge,
)
from docqna.exceptions import GenerationError
from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.models import (
    CitationSource,
    RAGResponse,
)
from docqna.rag.pipeline import RAGPipeline


class FakeJudgeLLM(BaseLLMProvider):
    """Deterministic judge provider for unit tests."""

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        self.generate_calls += 1

        assert "QUESTION:" in prompt
        assert "GENERATED ANSWER:" in prompt
        assert "RETRIEVED CONTEXT:" in prompt
        assert system_prompt is not None

        return """
```json
{
  "answer_relevance": 0.9,
  "context_relevance": 0.8,
  "faithfulness": 0.7,
  "citation_correctness": 0.6,
  "rationale": "Synthetic deterministic judge result."
}
```
""".strip()

    def health_check(self) -> bool:
        return True


class FakeRAGPipeline(RAGPipeline):
    """Deterministic pipeline for RAG-quality tests."""

    def __init__(
        self,
        responses: dict[
            str,
            RAGResponse | Exception
        ],
    ) -> None:
        self.responses = responses
        self.last_metadata_filters: (
            MetadataFilters | None
        ) = None

    def ask_with_citations(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> RAGResponse:
        self.last_metadata_filters = (
            metadata_filters
        )

        value = self.responses[
            query
        ]

        if isinstance(
            value,
            Exception,
        ):
            raise value

        return value


def answered_response() -> RAGResponse:
    return RAGResponse(
        answer=(
            "A supported answer based on "
            "the supplied evidence [1]."
        ),
        sources=[
            CitationSource(
                citation_id=1,
                chunk_id="chunk_1",
                rank=1,
                filename="sample.pdf",
                file_type="pdf",
                page_start=10,
                page_end=10,
                section=None,
                distance=-5.0,
                text=(
                    "The supplied evidence directly "
                    "supports the answer."
                ),
            )
        ],
        cited_source_ids=[1],
    )


def abstained_response() -> RAGResponse:
    return RAGResponse(
        answer=(
            "The provided documents do not contain enough "
            "information to answer the question."
        ),
        sources=[],
        cited_source_ids=[],
    )


def build_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="judged",
            question="answered_case",
            should_answer=True,
            relevant_pages=[10],
        ),
        EvaluationCase(
            case_id="abstained",
            question="abstained_case",
            should_answer=True,
            relevant_pages=[20],
        ),
        EvaluationCase(
            case_id="error",
            question="error_case",
            should_answer=True,
            relevant_pages=[30],
        ),
        EvaluationCase(
            case_id="unanswerable",
            question="unanswerable_case",
            should_answer=False,
        ),
    ]


def main() -> None:
    judge_llm = FakeJudgeLLM()

    judge = RAGQualityJudge(
        llm=judge_llm,
    )

    pipeline = FakeRAGPipeline(
        {
            "answered_case": (
                answered_response()
            ),
            "abstained_case": (
                abstained_response()
            ),
            "error_case": GenerationError(
                "Synthetic pipeline failure."
            ),
            "unanswerable_case": (
                abstained_response()
            ),
        }
    )

    evaluator = RAGQualityEvaluator(
        rag_pipeline=pipeline,
        judge=judge,
    )

    report = evaluator.evaluate(
        cases=build_cases(),
        top_k=5,
    )

    assert report.num_cases == 4
    assert report.answerable_cases == 3
    assert (
        report.skipped_unanswerable_cases
        == 1
    )

    assert report.judged_cases == 1
    assert report.abstained_cases == 1
    assert report.failed_cases == 1

    assert (
        report.mean_answer_relevance
        == 0.9
    )
    assert (
        report.mean_context_relevance
        == 0.8
    )
    assert (
        report.mean_faithfulness
        == 0.7
    )
    assert (
        report.mean_citation_correctness
        == 0.6
    )

    assert judge_llm.generate_calls == 1

    print(
        "LLM judge JSON parsing: PASS"
    )
    print(
        "Answer relevance metric: PASS"
    )
    print(
        "Context relevance metric: PASS"
    )
    print(
        "Faithfulness metric: PASS"
    )
    print(
        "Citation correctness metric: PASS"
    )
    print(
        "Unanswerable-case skipping: PASS"
    )
    print(
        "Abstention/error accounting: PASS"
    )

    filtered_report = evaluator.evaluate(
        cases=[
            EvaluationCase(
                case_id="filtered",
                question="answered_case",
                should_answer=True,
                relevant_pages=[10],
            )
        ],
        top_k=5,
        metadata_filters={
            "document_id": "doc_test",
        },
    )

    assert filtered_report.judged_cases == 1
    assert (
        pipeline.last_metadata_filters
        == {
            "document_id": "doc_test",
        }
    )

    print(
        "Metadata-filter propagation: PASS"
    )

    invalid_judge = FakeJudgeLLM()

    def invalid_generate(
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        return (
            '{"answer_relevance": 1.2, '
            '"context_relevance": 0.8, '
            '"faithfulness": 0.7, '
            '"citation_correctness": 0.6, '
            '"rationale": "invalid"}'
        )

    invalid_judge.generate = (
        invalid_generate
    )  # type: ignore[method-assign]

    invalid = RAGQualityJudge(
        llm=invalid_judge
    )

    try:
        invalid.score(
            question="Question?",
            response=answered_response(),
        )

    except ValueError:
        print(
            "Invalid judge score rejection: PASS"
        )

    else:
        raise AssertionError(
            "Out-of-range judge score "
            "was not rejected."
        )

    print()
    print(
        "Phase 21D RAG quality tests passed."
    )


if __name__ == "__main__":
    main()
