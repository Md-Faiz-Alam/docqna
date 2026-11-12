import re
import unicodedata

from docqna.rag.models import RAGResponse


def normalize_text(text: str) -> str:
    """
    Normalize text for lightweight lexical evaluation.

    The normalization:
    - applies Unicode NFKC
    - converts text to lowercase
    - replaces punctuation with spaces
    - collapses repeated whitespace
    """

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


def contains_expected_term(text: str, expected_term: str) -> bool:
    """
    Determine whether an expected answer concept
    is represented lexically in the supplied text.

    First tries normalized phrase matching.

    If the complete phrase is not contiguous,
    all normalized term tokens must appear in
    the text.
    """

    normalized_text = normalize_text(text)

    normalized_term = normalize_text(expected_term)

    if not normalized_term:
        return False

    if normalized_term in normalized_text:
        return True

    text_tokens = set(normalized_text.split())

    term_tokens = normalized_term.split()

    if not term_tokens:
        return False

    return all(token in text_tokens for token in term_tokens)


def matched_terms(text: str, expected_terms: list[str]) -> list[str]:
    """Return expected terms represented in text."""

    return [term for term in expected_terms if contains_expected_term(text, term)]


def term_coverage(text: str, expected_terms: list[str]) -> float:
    """
    Return the fraction of expected terms
    represented in text.
    """

    if not expected_terms:
        return 0.0

    matches = matched_terms(text, expected_terms)

    return len(matches) / len(expected_terms)


def extract_canonical_citation_ids(answer: str) -> list[int]:
    """Extract canonical citations such as [1] and [2]."""

    return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]


def citation_presence(response: RAGResponse) -> float:
    """
    Return 1 when the answer contains at least
    one canonical citation and structured cited
    source IDs are also present.
    """

    answer_ids = extract_canonical_citation_ids(response.answer)

    if answer_ids and response.cited_source_ids:
        return 1.0

    return 0.0


def citation_validity(response: RAGResponse) -> float:
    """
    Independently verify that answer citations:

    - use available source IDs
    - agree with structured cited_source_ids
    """

    answer_ids = extract_canonical_citation_ids(response.answer)

    if not answer_ids:
        return 0.0

    if not response.sources:
        return 0.0

    valid_source_ids = {source.citation_id for source in response.sources}

    if any(citation_id not in valid_source_ids for citation_id in answer_ids):
        return 0.0

    if set(answer_ids) != set(response.cited_source_ids):
        return 0.0

    return 1.0


def cited_evidence_text(response: RAGResponse) -> str:
    """
    Join the text of sources actually cited
    by the generated answer.
    """

    cited_ids = set(response.cited_source_ids)

    texts = [source.text for source in response.sources if source.citation_id in cited_ids]

    return "\n".join(texts)


def answer_word_count(answer: str) -> int:
    """Return approximate whitespace-based word count."""

    return len(answer.split())
