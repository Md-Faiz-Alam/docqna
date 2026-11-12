from docqna.evaluation import AnswerabilityEvaluator, EvaluationCase
from docqna.exceptions import GenerationError
from docqna.filters import MetadataFilters
from docqna.rag.models import RAGResponse
from docqna.rag.pipeline import RAGPipeline


class FakeRAGPipeline(RAGPipeline):
    """Deterministic RAG stand-in for answerability tests."""

    def __init__(self, responses: dict[str, RAGResponse | Exception]) -> None:
        self.responses = responses

    def ask_with_citations(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> RAGResponse:
        value = self.responses[query]

        if isinstance(value, Exception):
            raise value

        return value


def answered_response() -> RAGResponse:
    """Return a minimal non-abstaining response."""

    return RAGResponse(answer="Supported answer [1]", sources=[], cited_source_ids=[1])


def abstained_response() -> RAGResponse:
    """Return the production abstention shape."""

    return RAGResponse(
        answer=("The provided documents do not contain enough information to answer the question."),
        sources=[],
        cited_source_ids=[],
    )


def build_cases() -> list[EvaluationCase]:
    """Build all four answerability confusion-matrix outcomes."""

    return [
        EvaluationCase(
            case_id="true_answer",
            question="answerable_answered",
            should_answer=True,
            relevant_pages=[1],
        ),
        EvaluationCase(
            case_id="false_abstention",
            question="answerable_abstained",
            should_answer=True,
            relevant_pages=[2],
        ),
        EvaluationCase(
            case_id="false_answer", question="unanswerable_answered", should_answer=False
        ),
        EvaluationCase(
            case_id="true_abstention", question="unanswerable_abstained", should_answer=False
        ),
        EvaluationCase(case_id="error_case", question="pipeline_error", should_answer=False),
    ]


def test_answerability_metrics() -> None:
    pipeline = FakeRAGPipeline(
        {
            "answerable_answered": (answered_response()),
            "answerable_abstained": (abstained_response()),
            "unanswerable_answered": (answered_response()),
            "unanswerable_abstained": (abstained_response()),
            "pipeline_error": GenerationError("Synthetic generation failure."),
        }
    )

    evaluator = AnswerabilityEvaluator(rag_pipeline=pipeline)

    report = evaluator.evaluate(cases=build_cases(), top_k=5)

    assert report.num_cases == 5
    assert report.successful_cases == 4
    assert report.error_cases == 1

    assert report.true_answers == 1
    assert report.true_abstentions == 1
    assert report.false_answers == 1
    assert report.false_abstentions == 1

    assert report.accuracy == 0.5

    assert report.answer_precision == 0.5
    assert report.answer_recall == 0.5
    assert report.answer_f1 == 0.5

    assert report.abstention_precision == 0.5
    assert report.abstention_recall == 0.5
    assert report.abstention_f1 == 0.5

    filtered_report = evaluator.evaluate(
        cases=[
            EvaluationCase(
                case_id="filtered", question="unanswerable_abstained", should_answer=False
            )
        ],
        top_k=5,
        metadata_filters={"document_id": "doc_test"},
    )

    assert filtered_report.true_abstentions == 1
