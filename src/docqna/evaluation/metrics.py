import math

from docqna.retrieval.models import RetrievalResult


def result_pages(
    result: RetrievalResult,
) -> set[int]:
    """Return all page numbers covered by a result."""

    if result.page_start is None:
        return set()

    page_end = (
        result.page_end
        if result.page_end is not None
        else result.page_start
    )

    return set(
        range(
            result.page_start,
            page_end + 1,
        )
    )


def is_relevant(
    result: RetrievalResult,
    relevant_pages: set[int],
    relevant_chunk_ids: set[str] | None = None,
) -> bool:
    """
    Return True when a result matches labeled retrieval evidence.

    Chunk-ID labels take precedence when supplied. Otherwise,
    relevance is determined by overlap with labeled source pages.
    """

    if relevant_chunk_ids:
        return (
            result.chunk_id
            in relevant_chunk_ids
        )

    return bool(
        result_pages(result)
        & relevant_pages
    )


def hit_at_k(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """Return 1 when at least one relevant result appears in top-k."""

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    for result in results[:k]:
        if is_relevant(
            result,
            relevant_pages,
            relevant_chunk_ids,
        ):
            return 1.0

    return 0.0


def reciprocal_rank(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """Return reciprocal rank of the first relevant result."""

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    for rank, result in enumerate(
        results[:k],
        start=1,
    ):
        if is_relevant(
            result,
            relevant_pages,
            relevant_chunk_ids,
        ):
            return 1.0 / rank

    return 0.0


def page_recall_at_k(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
) -> float:
    """Measure how many labeled relevant pages were retrieved."""

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    if not relevant_pages:
        return 0.0

    retrieved_pages: set[int] = set()

    for result in results[:k]:
        retrieved_pages.update(
            result_pages(result)
        )

    covered_pages = (
        retrieved_pages
        & relevant_pages
    )

    return (
        len(covered_pages)
        / len(relevant_pages)
    )


def precision_at_k(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """
    Return the fraction of top-k result slots that are relevant.

    The denominator is always k, so a retriever returning fewer
    than k results is appropriately penalized.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    relevant_count = sum(
        1
        for result in results[:k]
        if is_relevant(
            result,
            relevant_pages,
            relevant_chunk_ids,
        )
    )

    return relevant_count / k


def recall_at_k(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """
    Return retrieval recall at k.

    When explicit relevant chunk IDs are available, chunk recall
    is used. Otherwise the current page-level ground truth is used,
    making this equivalent to labeled-page recall.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    if relevant_chunk_ids:
        retrieved_chunk_ids = {
            result.chunk_id
            for result in results[:k]
        }

        matched_chunk_ids = (
            retrieved_chunk_ids
            & relevant_chunk_ids
        )

        return (
            len(matched_chunk_ids)
            / len(relevant_chunk_ids)
        )

    return page_recall_at_k(
        results=results,
        relevant_pages=relevant_pages,
        k=k,
    )


def ndcg_at_k(
    results: list[RetrievalResult],
    relevant_pages: set[int],
    k: int,
    relevant_chunk_ids: set[str] | None = None,
) -> float:
    """
    Return binary nDCG@k for retrieval ranking quality.

    With explicit chunk-level qrels, the ideal ranking uses the
    number of labeled relevant chunks.

    With page-level qrels, the dataset does not enumerate every
    relevant chunk in the corpus. In that case nDCG normalizes
    against the number of relevant results observed in top-k.
    Recall@k separately measures whether the labeled pages were
    actually covered.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    relevance = [
        1.0
        if is_relevant(
            result,
            relevant_pages,
            relevant_chunk_ids,
        )
        else 0.0
        for result in results[:k]
    ]

    dcg = sum(
        gain
        / math.log2(rank + 1)
        for rank, gain in enumerate(
            relevance,
            start=1,
        )
    )

    if relevant_chunk_ids:
        ideal_relevant_count = min(
            k,
            len(relevant_chunk_ids),
        )
    else:
        ideal_relevant_count = int(
            sum(relevance)
        )

    if ideal_relevant_count <= 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    return dcg / idcg
