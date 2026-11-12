from docqna.retrieval.models import RetrievalResult

ABSTENTION_ANSWER = (
    "The provided documents do not contain enough information to answer the question."
)


def _build_cited_context(results: list[RetrievalResult]) -> str:
    """Build numbered source blocks for cited RAG prompts."""

    context_parts: list[str] = []

    for citation_id, result in enumerate(results, start=1):
        source_parts = [f"File: {result.filename}"]

        if result.page_start is not None:
            if result.page_end is not None and result.page_end != result.page_start:
                source_parts.append(f"Pages: {result.page_start}-{result.page_end}")
            else:
                source_parts.append(f"Page: {result.page_start}")

        if result.section is not None:
            source_parts.append(f"Section: {result.section}")

        source_description = " | ".join(source_parts)

        context_parts.append(f"SOURCE [{citation_id}]\n{source_description}\n{result.text}")

    return "\n\n".join(context_parts)


def _build_grounding_rules(source_count: int) -> str:
    """
    Build concise grounding rules that small local models
    can follow reliably.
    """

    return (
        "INSTRUCTIONS:\n"
        "- Answer only from the document sources below. "
        "Do not use outside knowledge.\n"
        "- Every factual sentence must end with at least one "
        "supporting citation such as [1].\n"
        "- Choose the citation by checking which source actually "
        "contains the evidence. Do not assign citation numbers "
        "sequentially just because sentences appear in sequence.\n"
        "- A single source may support several claims. Reuse the "
        "same citation when appropriate.\n"
        f"- Use only source IDs [1] through [{source_count}].\n"
        "- Return only the answer. Do not repeat the prompt, "
        "instructions, source blocks, or user question.\n"
        "- Do not discuss unused, rejected, or irrelevant "
        "sources.\n"
        "- Do not explain how citations were selected or "
        "why a source was not cited.\n"
        "- If the sources do not contain enough information, "
        "return exactly:\n"
        f"{ABSTENTION_ANSWER}"
    )


def build_cited_rag_prompt(query: str, results: list[RetrievalResult]) -> str:
    """
    Build a compact cited prompt optimized for small local LLMs.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not results:
        raise ValueError("Retrieval results cannot be empty.")

    context = _build_cited_context(results)

    rules = _build_grounding_rules(len(results))

    return (
        "You are a document question-answering assistant.\n"
        "Give a concise answer of at most 120 words.\n\n"
        f"{rules}\n\n"
        "Example citation style:\n"
        "Gantt charts visualize project schedules [1]. "
        "Three-point estimating uses multiple duration "
        "estimates [2].\n\n"
        "DOCUMENT SOURCES:\n\n"
        f"{context}\n\n"
        "USER QUESTION:\n"
        f"{query}\n\n"
        "Remember: every factual sentence in the answer must "
        "contain a supporting [n] citation.\n"
        "FINAL ANSWER:\n"
    )


def build_citation_repair_prompt(
    query: str, results: list[RetrievalResult], candidate_answer: str
) -> str:
    """
    Build one compact citation-repair prompt.

    The repair call is intentionally shorter than the main
    prompt so a small local model can focus on citation format.
    """

    if not candidate_answer.strip():
        candidate_answer = "(empty answer)"

    context = _build_cited_context(results)

    return (
        "The previous answer failed citation validation because "
        "its citations were missing or invalid.\n"
        "Rewrite the answer using only the document sources below.\n"
        "Every factual sentence must end with the source number "
        "that directly supports it, for example [1].\n"
        "Do not assign numbers sequentially. Reuse the same "
        "source number when it supports multiple claims.\n"
        f"Use only source IDs [1] through [{len(results)}].\n"
        "If the sources do not support the answer, return exactly:\n"
        f"{ABSTENTION_ANSWER}\n"
        "Return only the corrected answer.\n\n"
        "PREVIOUS ANSWER:\n"
        f"{candidate_answer}\n\n"
        "DOCUMENT SOURCES:\n\n"
        f"{context}\n\n"
        "USER QUESTION:\n"
        f"{query}\n\n"
        "CORRECTED FINAL ANSWER:\n"
    )
