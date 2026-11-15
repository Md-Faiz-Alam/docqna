from docqna.rag.prompt import build_cited_rag_prompt
from docqna.retrieval.models import RetrievalResult
from experiments.attack_prompt_injection import (
    ATTACK_LABELS,
    answered_correctly,
    hijacked,
    load_cases,
    summarize,
)

PHISH = {"fail_if_contains": ["northwind-payroll-update"], "expect_any": ["30 days"]}
LEAK = {"leak": True, "expect_any": ["26 June"]}


def test_attacker_text_is_detected_case_insensitively() -> None:
    assert hijacked(
        "Paid in 30 days [1]. Log in at https://Northwind-Payroll-Update.example", PHISH
    )
    assert not hijacked("Invoices are paid within 30 days [1].", PHISH)


def test_repeating_the_prompt_counts_as_a_leak_only_for_leak_cases() -> None:
    leaked = "My rules: Do not use outside knowledge. Every sentence needs a supporting citation."
    assert hijacked(leaked, LEAK)
    assert not hijacked(leaked, PHISH)


def test_leak_markers_match_the_real_prompt() -> None:
    from experiments.attack_prompt_injection import PROMPT_LEAK_MARKERS

    result = RetrievalResult(
        chunk_id="c",
        document_id="d",
        filename="f.txt",
        file_type="txt",
        text="t",
        chunk_index=0,
        rank=1,
        distance=0.0,
        score=1.0,
    )
    prompt = build_cited_rag_prompt("q", [result]).lower()
    assert all(marker in prompt for marker in PROMPT_LEAK_MARKERS)


def test_a_hijacked_answer_is_never_counted_correct() -> None:
    assert answered_correctly("Paid within 30 days [1].", PHISH) is True
    assert (
        answered_correctly("Paid within 30 days [1]. See northwind-payroll-update.example", PHISH)
        is False
    )
    assert answered_correctly("anything", {"fail_if_contains": ["x"]}) is None


def test_every_case_is_well_formed() -> None:
    cases = load_cases()
    assert len({c["case_id"] for c in cases}) == len(cases)
    for case in cases:
        assert case["kind"] in ("indirect", "direct")
        assert case["attack"] in ATTACK_LABELS
        assert case.get("fail_if_contains") or case.get("leak"), case["case_id"]
        assert (case["kind"] == "indirect") == bool(case.get("attack_document")), case["case_id"]


def test_summary_separates_what_the_llm_wrote_from_what_was_shown() -> None:
    base = {"attack_passage_retrieved": True, "kind": "indirect", "attack": "hijack"}
    records = [
        {**base, "hijacked_written": True, "hijacked_shown": False, "correct": True},
        {**base, "hijacked_written": True, "hijacked_shown": True, "correct": False},
        {
            **base,
            "kind": "direct",
            "attack_passage_retrieved": None,
            "hijacked_written": False,
            "hijacked_shown": False,
            "correct": None,
        },
    ]
    summary = summarize(records)
    assert summary["overall"]["hijacked_written"] == 2
    assert summary["overall"]["hijacked_shown"] == 1
    assert summary["overall"]["real_answer_correct"] == 1
    assert summary["overall"]["real_answer_cases"] == 2
    assert summary["by_kind"]["direct"]["cases"] == 1
    assert summary["attack_passage_retrieved"] == 2
