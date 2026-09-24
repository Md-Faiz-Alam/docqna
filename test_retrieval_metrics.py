from docqna.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    page_recall_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from docqna.retrieval.models import RetrievalResult


def make_result(
    *,
    chunk_id: str,
    page_start: int | None,
    page_end: int | None = None,
) -> RetrievalResult:
    """
    Build the minimal typed RetrievalResult needed by metric tests.

    model_construct() intentionally skips full Pydantic validation
    because these unit tests exercise only chunk/page relevance logic.
    """

    return RetrievalResult.model_construct(
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
    )


def main() -> None:
    relevant_pages = {
        10,
        11,
    }

    results = [
        make_result(
            chunk_id="irrelevant_1",
            page_start=2,
        ),
        make_result(
            chunk_id="relevant_a",
            page_start=10,
        ),
        make_result(
            chunk_id="irrelevant_2",
            page_start=7,
        ),
        make_result(
            chunk_id="relevant_b",
            page_start=11,
        ),
        make_result(
            chunk_id="irrelevant_3",
            page_start=20,
        ),
    ]

    # Existing metrics remain correct.
    assert hit_at_k(
        results,
        relevant_pages,
        1,
    ) == 0.0

    assert hit_at_k(
        results,
        relevant_pages,
        3,
    ) == 1.0

    assert reciprocal_rank(
        results,
        relevant_pages,
        5,
    ) == 0.5

    assert page_recall_at_k(
        results,
        relevant_pages,
        5,
    ) == 1.0

    print(
        "Existing retrieval metrics: PASS"
    )

    # 2 of 5 result slots overlap relevant pages.
    precision = precision_at_k(
        results,
        relevant_pages,
        5,
    )

    assert abs(
        precision - 0.4
    ) < 1e-12

    print(
        "Precision@K: PASS"
    )

    # Both labeled relevant pages were covered.
    recall = recall_at_k(
        results,
        relevant_pages,
        5,
    )

    assert recall == 1.0

    print(
        "Recall@K: PASS"
    )

    # Relevant results are ranks 2 and 4, so nDCG must
    # be positive but below the ideal ordering.
    ndcg = ndcg_at_k(
        results,
        relevant_pages,
        5,
    )

    assert 0.0 < ndcg < 1.0

    print(
        "nDCG@K: PASS"
    )

    # Explicit chunk qrels take precedence over page labels.
    chunk_relevant = {
        "relevant_a",
        "missing_relevant_chunk",
    }

    chunk_recall = recall_at_k(
        results,
        relevant_pages=set(),
        k=5,
        relevant_chunk_ids=chunk_relevant,
    )

    assert chunk_recall == 0.5

    chunk_precision = precision_at_k(
        results,
        relevant_pages=set(),
        k=5,
        relevant_chunk_ids=chunk_relevant,
    )

    assert abs(
        chunk_precision - 0.2
    ) < 1e-12

    chunk_ndcg = ndcg_at_k(
        results,
        relevant_pages=set(),
        k=5,
        relevant_chunk_ids=chunk_relevant,
    )

    assert 0.0 < chunk_ndcg < 1.0

    print(
        "Chunk-level qrel support: PASS"
    )

    # Invalid K values must fail explicitly.
    try:
        precision_at_k(
            results,
            relevant_pages,
            0,
        )
    except ValueError:
        print(
            "Invalid K rejection: PASS"
        )
    else:
        raise AssertionError(
            "k=0 was not rejected."
        )

    print()
    print(
        "Phase 21B retrieval metric tests passed."
    )


if __name__ == "__main__":
    main()
