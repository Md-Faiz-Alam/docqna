import json

from docqna.rag import ABSTENTION_ANSWER
from experiments.build_heldout_set import HELDOUT_DIR
from experiments.evaluate_heldout import contains_answer, normalize, refused_by, summarize


def test_squad_normalisation() -> None:
    assert normalize("The Normans, in 1066!") == "normans in 1066"
    assert normalize("Financial crisis of 2007–08") == "financial crisis of 200708"


def test_answer_matching_is_whole_word_and_ignores_case_and_articles() -> None:
    assert contains_answer("They arrived in 1066 [1].", ["1066"])
    assert contains_answer(
        "It was the Financial Crisis of 2007–08 [2].", ["Financial crisis of 2007–08"]
    )
    assert not contains_answer("They arrived in 10660 [1].", ["1066"])
    assert not contains_answer("anything", [])


def _record(**overrides: object) -> dict[str, object]:
    base = {
        "should_answer": True,
        "answered": True,
        "correct": True,
        "llm_called": True,
        "written": "Answer [1].",
        "evidence_in_top_k": True,
    }
    base.update(overrides)
    base["refused_by"] = refused_by(base)
    return base


def test_refusals_are_attributed_to_the_right_layer() -> None:
    assert _record()["refused_by"] is None
    assert _record(answered=False, llm_called=False)["refused_by"] == "retrieval_gate"
    assert _record(answered=False, written=ABSTENTION_ANSWER)["refused_by"] == "llm"
    assert _record(answered=False)["refused_by"] == "check_after_writing"


def test_summary_counts() -> None:
    records = [
        _record(),
        _record(correct=False),
        _record(answered=False, correct=False, llm_called=False, evidence_in_top_k=False),
        _record(should_answer=False, answered=False, correct=True, written=ABSTENTION_ANSWER),
        _record(should_answer=False, answered=True, correct=False),
    ]
    s = summarize(records)

    assert (s["correct"], s["wrong"], s["declined_answerable"]) == (1, 1, 1)
    assert (s["refused_unanswerable"], s["made_up"]) == (1, 1)
    assert s["evidence_in_top_k"] == 2
    assert s["refusals_answerable"]["retrieval_gate"] == 1
    assert s["refusals_unanswerable"]["llm"] == 1


def test_committed_heldout_set_is_balanced_and_well_formed() -> None:
    payload = json.loads((HELDOUT_DIR / "cases.json").read_text("utf-8"))
    cases = payload["cases"]
    documents = {p.name for p in (HELDOUT_DIR / "corpus").glob("*.txt")}

    assert len(cases) == 100
    assert len({c["case_id"] for c in cases}) == 100
    assert sum(c["should_answer"] for c in cases) == 50
    assert {c["document"] for c in cases} == documents
    assert all(c["answers"] for c in cases if c["should_answer"])
    assert "CC BY-SA 4.0" in payload["license"]
