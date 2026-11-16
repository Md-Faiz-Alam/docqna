import re

import pytest

from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.pipeline import RAGPipeline
from docqna.rag.prompt import (
    build_citation_repair_prompt,
    build_cited_rag_prompt,
    build_separated_rag_prompt,
)
from docqna.retrieval.base import BaseRetriever
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


# --- Rules apart from documents (SEPARATE_PROMPT_ROLES) ----------------------


def test_separated_prompt_keeps_rules_out_of_the_user_message() -> None:
    results = [make_result(1, "Leave is 24 days."), make_result(2, "Hours are 38.")]

    system, user = build_separated_rag_prompt("How much leave?", results, "ab12cd34")

    assert "INSTRUCTIONS:" in system and "INSTRUCTIONS:" not in user
    assert "Leave is 24 days." in user and "Leave is 24 days." not in system
    assert "Use only source IDs [1] through [2]" in system
    assert '<source-ab12cd34 id="1" file="sample.pdf" pages="32-33">' in user
    assert user.count("</source-ab12cd34>") == 2
    assert user.index("DOCUMENT SOURCES:") < user.index("USER QUESTION:")
    assert user.rstrip().endswith("FINAL ANSWER:")


def test_separated_prompt_rejects_unsafe_nonce_and_empty_input() -> None:
    results = [make_result(1, "Text.")]

    for nonce in ("", "a>b", "x y"):
        with pytest.raises(ValueError):
            build_separated_rag_prompt("q", results, nonce)
    with pytest.raises(ValueError):
        build_separated_rag_prompt(" ", results, "ab12")
    with pytest.raises(ValueError):
        build_separated_rag_prompt("q", [], "ab12")


class RecordingLLM(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return "Leave is 24 days [1]."

    def health_check(self) -> bool:
        return True


class OneResult(BaseRetriever):
    def retrieve(self, query, top_k=5, metadata_filters=None):
        return [make_result(1, "Leave is 24 days.")]


def test_pipeline_sends_system_prompt_and_fresh_tag_only_when_enabled() -> None:
    combined, separated = RecordingLLM(), RecordingLLM()
    RAGPipeline(OneResult(), combined).ask_with_citations("Leave?")
    pipeline = RAGPipeline(OneResult(), separated, separate_prompt_roles=True)
    pipeline.ask_with_citations("Leave?")
    list(pipeline.stream_with_citations("Leave?"))

    assert combined.calls[0][1] is None
    assert "INSTRUCTIONS:" in combined.calls[0][0]

    tags = set()
    for prompt, system in separated.calls:
        assert system is not None and "INSTRUCTIONS:" in system
        assert "INSTRUCTIONS:" not in prompt
        tags.add(re.search(r"<(source-[0-9a-f]+) ", prompt).group(1))  # type: ignore[union-attr]
    assert len(tags) == 2  # a new random tag name for every request
