from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.hybrid import RRFHybridRetriever
from docqna.retrieval.models import RetrievalResult


def make_result(
    chunk_id: str,
    rank: int,
    chunk_index: int,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        rank=rank,
        distance=0.0,
        filename="sample.pdf",
        file_type="pdf",
        chunk_index=chunk_index,
    )


class StubRetriever(BaseRetriever):
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return self.results[:top_k]


dense = StubRetriever(
    [
        make_result("chunk_a", 1, 0),
        make_result("chunk_b", 2, 1),
        make_result("chunk_c", 3, 2),
    ]
)

bm25 = StubRetriever(
    [
        make_result("chunk_b", 1, 1),
        make_result("chunk_d", 2, 3),
        make_result("chunk_a", 3, 0),
    ]
)


retriever = RRFHybridRetriever(
    dense_retriever=dense,
    sparse_retriever=bm25,
    rrf_k=60,
    candidate_pool_size=3,
)


results = retriever.retrieve(
    query="test query",
    top_k=3,
)


assert len(results) == 3

# chunk_a and chunk_b occur in both rankings,
# so they should outrank single-source chunks.
assert {
    results[0].chunk_id,
    results[1].chunk_id,
} == {
    "chunk_a",
    "chunk_b",
}

assert (
    results[0].retrieval_method
    == "hybrid_rrf"
)

assert results[0].score is not None
assert results[0].score > 0

assert results[0].rank == 1
assert results[1].rank == 2
assert results[2].rank == 3


print("Hybrid results:")

for result in results:
    print(
        result.rank,
        result.chunk_id,
        "RRF score=",
        round(
            result.score or 0.0,
            6,
        ),
    )


print(
    "\nRRF hybrid retrieval test passed"
)