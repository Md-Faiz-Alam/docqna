from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.pipeline import (
    ABSTENTION_ANSWER,
    RAGPipeline,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import (
    RetrievalResult,
)


class FakeRetriever(BaseRetriever):
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.last_metadata_filters: MetadataFilters | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        self.last_metadata_filters = (
            metadata_filters
        )

        return self.results[
            :top_k
        ]


class FakeLLM(BaseLLMProvider):
    def __init__(
        self,
        responses: list[str],
    ) -> None:
        self.responses = list(
            responses
        )

        self.prompts: list[
            str
        ] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        self.prompts.append(
            prompt
        )

        if not self.responses:
            raise AssertionError(
                "FakeLLM has no response left."
            )

        return self.responses.pop(
            0
        )

    def health_check(self) -> bool:
        return True


def make_result(
    *,
    rank: int = 1,
    document_id: str = "doc_test",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=(
            f"chunk_{rank}"
        ),
        text=(
            "Critical Path Method uses a single "
            "deterministic task-duration estimate. "
            "PERT uses optimistic, most likely, and "
            "pessimistic estimates."
        ),
        rank=rank,
        distance=0.1,
        score=5.0,
        retrieval_method=(
            "cross_encoder_rerank"
        ),
        filename="sample.pdf",
        file_type="pdf",
        chunk_index=rank - 1,
        document_id=document_id,
        document_version=1,
        page_start=38,
        page_end=39,
    )


def make_pipeline(
    responses: list[str],
) -> tuple[
    RAGPipeline,
    FakeRetriever,
    FakeLLM,
]:
    retriever = FakeRetriever(
        [
            make_result(),
        ]
    )

    llm = FakeLLM(
        responses
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
    )

    return (
        pipeline,
        retriever,
        llm,
    )


def test_valid_cited_answer() -> None:
    pipeline, _, llm = (
        make_pipeline(
            [
                (
                    "CPM uses a deterministic "
                    "estimate [1]."
                ),
            ]
        )
    )

    response = (
        pipeline.ask_with_citations(
            "How does CPM estimate duration?",
            top_k=1,
        )
    )

    assert response.cited_source_ids == [
        1
    ]

    assert len(
        response.sources
    ) == 1

    assert len(
        llm.prompts
    ) == 1

    prompt = llm.prompts[
        0
    ]

    assert (
        "Do not use outside knowledge"
        in prompt
    )

    assert (
        ABSTENTION_ANSWER
        in prompt
    )

    print(
        "Valid grounded answer: PASS"
    )


def test_missing_citation_repair() -> None:
    pipeline, _, llm = (
        make_pipeline(
            [
                (
                    "CPM uses a deterministic "
                    "estimate."
                ),
                (
                    "CPM uses a deterministic "
                    "estimate [1]."
                ),
            ]
        )
    )

    response = (
        pipeline.ask_with_citations(
            "How does CPM estimate duration?",
            top_k=1,
        )
    )

    assert response.cited_source_ids == [
        1
    ]

    assert len(
        llm.prompts
    ) == 2

    assert (
        "failed citation validation"
        in llm.prompts[1]
    )

    print(
        "Missing citation repair: PASS"
    )


def test_invalid_citation_repair() -> None:
    pipeline, _, llm = (
        make_pipeline(
            [
                (
                    "CPM uses a deterministic "
                    "estimate [99]."
                ),
                (
                    "CPM uses a deterministic "
                    "estimate [1]."
                ),
            ]
        )
    )

    response = (
        pipeline.ask_with_citations(
            "How does CPM estimate duration?",
            top_k=1,
        )
    )

    assert response.cited_source_ids == [
        1
    ]

    assert len(
        llm.prompts
    ) == 2

    print(
        "Invalid citation repair: PASS"
    )


def test_failed_repair_abstains() -> None:
    pipeline, _, llm = (
        make_pipeline(
            [
                "Unsupported answer.",
                (
                    "Still unsupported and "
                    "still uncited."
                ),
            ]
        )
    )

    response = (
        pipeline.ask_with_citations(
            "How does CPM estimate duration?",
            top_k=1,
        )
    )

    assert (
        response.answer
        == ABSTENTION_ANSWER
    )

    assert response.sources == []
    assert (
        response.cited_source_ids
        == []
    )

    assert len(
        llm.prompts
    ) == 2

    print(
        "Failed repair safe abstention: PASS"
    )


def test_model_abstention_is_valid() -> None:
    pipeline, _, llm = (
        make_pipeline(
            [
                ABSTENTION_ANSWER,
            ]
        )
    )

    response = (
        pipeline.ask_with_citations(
            "Unsupported question?",
            top_k=1,
        )
    )

    assert (
        response.answer
        == ABSTENTION_ANSWER
    )

    assert response.sources == []
    assert (
        response.cited_source_ids
        == []
    )

    assert len(
        llm.prompts
    ) == 1

    print(
        "Model-side grounded abstention: PASS"
    )


def test_document_filter_preserved() -> None:
    pipeline, retriever, _ = (
        make_pipeline(
            [
                (
                    "CPM uses a deterministic "
                    "estimate [1]."
                ),
            ]
        )
    )

    pipeline.ask_with_citations(
        "How does CPM estimate duration?",
        top_k=1,
        metadata_filters={
            "document_id": (
                "doc_test"
            ),
        },
    )

    assert (
        retriever.last_metadata_filters
        == {
            "document_id": (
                "doc_test"
            ),
        }
    )

    print(
        "Document filter preservation: PASS"
    )


def test_citation_normalization() -> None:
    assert (
        RAGPipeline._normalize_citations(
            "Supported claim (Source: 1)."
        )
        == "Supported claim [1]."
    )

    assert (
        RAGPipeline._normalize_citations(
            "Supported claim [Citation 1]."
        )
        == "Supported claim [1]."
    )

    print(
        "Citation normalization: PASS"
    )


def main() -> None:
    test_valid_cited_answer()
    test_missing_citation_repair()
    test_invalid_citation_repair()
    test_failed_repair_abstains()
    test_model_abstention_is_valid()
    test_document_filter_preserved()
    test_citation_normalization()

    print()
    print(
        "RAG reliability hardening tests passed."
    )


if __name__ == "__main__":
    main()
