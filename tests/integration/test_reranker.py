import pytest

from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from docqna.retrieval.reranker import CrossEncoderReranker


@pytest.mark.integration
def test_reranker() -> None:
    def make_result(chunk_id: str, text: str, rank: int, chunk_index: int) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            text=text,
            rank=rank,
            distance=0.0,
            filename="sample.pdf",
            file_type="pdf",
            chunk_index=chunk_index,
        )

    class StubRetriever(BaseRetriever):
        def __init__(self, results: list[RetrievalResult]) -> None:
            self.results = results

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            return self.results[:top_k]

    candidates = [
        make_result(
            chunk_id="chunk_irrelevant_1",
            text=(
                "Project reports should include an executive "
                "summary, findings, conclusions, and appendices."
            ),
            rank=1,
            chunk_index=0,
        ),
        make_result(
            chunk_id="chunk_relevant",
            text=(
                "Earned Value Management uses Planned Value, "
                "Actual Cost, and Earned Value to measure "
                "project performance."
            ),
            rank=2,
            chunk_index=1,
        ),
        make_result(
            chunk_id="chunk_irrelevant_2",
            text=(
                "Risk identification techniques include "
                "brainstorming, checklists, and SWOT analysis."
            ),
            rank=3,
            chunk_index=2,
        ),
    ]

    base_retriever = StubRetriever(results=candidates)

    reranker = CrossEncoderReranker(
        base_retriever=base_retriever,
        model_name=("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        candidate_pool_size=3,
        device="cpu",
    )

    results = reranker.retrieve(
        query=("What metrics are used in Earned Value Management?"), top_k=3
    )

    assert len(results) == 3

    assert results[0].chunk_id == "chunk_relevant"

    assert results[0].retrieval_method == "cross_encoder_rerank"

    assert results[0].score is not None

    assert results[0].rank == 1
