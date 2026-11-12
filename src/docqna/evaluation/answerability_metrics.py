from docqna.rag.models import RAGResponse


def response_abstained(response: RAGResponse) -> bool:
    """
    Return True when the RAG pipeline abstained.

    The production pipeline represents abstention with no
    structured sources and no cited source IDs.
    """

    return not response.sources and not response.cited_source_ids


def safe_divide(numerator: int, denominator: int) -> float:
    """Return a safe floating-point ratio."""

    if denominator <= 0:
        return 0.0

    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    """Return harmonic mean of precision and recall."""

    denominator = precision + recall

    if denominator <= 0.0:
        return 0.0

    return 2.0 * precision * recall / denominator
