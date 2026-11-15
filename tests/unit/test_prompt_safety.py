from docqna.rag.pipeline import RAGPipeline
from docqna.rag.prompt import build_citation_repair_prompt, build_cited_rag_prompt
from docqna.retrieval.models import RetrievalResult


def make_result(rank: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk_{rank}",
        text=text,
        rank=rank,
        distance=0.1,
        score=5.0,
        retrieval_method="cross_encoder_rerank",
        filename="sample.pdf",
        file_type="pdf",
        chunk_index=rank - 1,
        document_id="doc_test",
        document_version=1,
        page_start=32,
        page_end=33,
    )


def test_prompt_ordering() -> None:
    results = [
        make_result(1, ("Responsibility, Respect, Fairness, and Honesty are ethical principles."))
    ]

    prompt = build_cited_rag_prompt(query="What are the four pillars?", results=results)

    rules_position = prompt.index("INSTRUCTIONS:")
    sources_position = prompt.index("DOCUMENT SOURCES:")
    question_position = prompt.index("USER QUESTION:")
    answer_position = prompt.index("FINAL ANSWER:")

    assert rules_position < sources_position < question_position < answer_position

    assert prompt.rstrip().endswith("FINAL ANSWER:")


def test_citation_mapping_instruction() -> None:
    results = [make_result(1, "One source can support several claims.")]

    prompt = build_cited_rag_prompt(query="Test?", results=results)

    assert "Do not assign citation numbers sequentially" in prompt

    assert "A single source may support several claims" in prompt

    assert "Do not discuss unused, rejected, or irrelevant sources" in prompt

    assert "Do not explain how citations were selected" in prompt


def test_prompt_leak_sanitizer() -> None:
    leaked = "CPM supports scheduling [1].\n\nUSER QUESTION:\nInternal prompt content"

    cleaned = RAGPipeline._sanitize_generated_answer(leaked)

    assert cleaned == ("CPM supports scheduling [1].")

    prefixed = "FINAL ANSWER:\nPERT uses three estimates [1]."

    cleaned_prefixed = RAGPipeline._sanitize_generated_answer(prefixed)

    assert cleaned_prefixed == ("PERT uses three estimates [1].")

    simplified_leak = (
        "Gantt charts support scheduling [1][2][3].\n\n"
        "INSTRUCTIONS:\n"
        "- Use only source IDs [1] through [5]."
    )

    cleaned_simplified = RAGPipeline._sanitize_generated_answer(simplified_leak)

    assert cleaned_simplified == ("Gantt charts support scheduling [1][2][3].")

    assert RAGPipeline._extract_citation_ids(cleaned_simplified) == [1, 2, 3]


def test_repair_prompt_ordering() -> None:
    results = [make_result(1, "PERT uses three time estimates.")]

    prompt = build_citation_repair_prompt(
        query="How does PERT estimate time?",
        results=results,
        candidate_answer=("PERT uses three estimates."),
    )

    assert "failed citation validation" in prompt

    assert (
        prompt.index("DOCUMENT SOURCES:")
        < prompt.index("USER QUESTION:")
        < prompt.index("CORRECTED FINAL ANSWER:")
    )

    assert prompt.rstrip().endswith("CORRECTED FINAL ANSWER:")
