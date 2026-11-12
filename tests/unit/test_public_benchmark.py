"""Fast checks that keep the public benchmark itself trustworthy (no models)."""

import re

import pytest

from docqna.evaluation.public_benchmark import (
    BENCHMARK_DIR,
    BenchmarkCase,
    is_relevant,
    load_cases,
    load_thresholds,
    score_cases,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.confidence import RetrievalConfidenceGate
from docqna.retrieval.models import RetrievalResult


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def test_every_evidence_phrase_exists_verbatim_in_its_document() -> None:
    corpus = {
        path.name: _normalise(path.read_text(encoding="utf-8"))
        for path in (BENCHMARK_DIR / "corpus").glob("*.txt")
    }

    for case in load_cases():
        if not case.should_answer:
            continue
        assert case.document in corpus, case.case_id
        for phrase in case.evidence:
            assert _normalise(phrase) in corpus[case.document], (case.case_id, phrase)


def test_benchmark_is_balanced_and_thresholds_are_valid() -> None:
    cases = load_cases()
    answerable = [c for c in cases if c.should_answer]

    assert len(answerable) >= 50
    assert len(cases) - len(answerable) >= 15
    assert len({c.document for c in answerable}) >= 4
    thresholds = load_thresholds()
    assert all(0 < value <= 1 for value in thresholds.model_dump().values())


def test_case_schema_rejects_inconsistent_labels() -> None:
    with pytest.raises(ValueError):
        BenchmarkCase(case_id="x", question="q?", should_answer=True)
    with pytest.raises(ValueError):
        BenchmarkCase(case_id="y", question="q?", should_answer=False, document="a.txt")


def result(rank: int, text: str, filename: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c{rank}",
        text=text,
        rank=rank,
        distance=-score,
        score=score,
        filename=filename,
        file_type="txt",
        chunk_index=rank,
    )


class Scripted(BaseRetriever):
    def __init__(self, by_question: dict[str, list[RetrievalResult]]) -> None:
        self.by_question = by_question

    def retrieve(self, query, top_k=5, metadata_filters=None):
        return self.by_question[query]


def test_scoring_computes_hits_mrr_and_gate_rates() -> None:
    cases = [
        BenchmarkCase(case_id="a", question="qa", document="d.txt", evidence=["Red  Fox"]),
        BenchmarkCase(case_id="b", question="qb", document="d.txt", evidence=["blue whale"]),
        BenchmarkCase(case_id="u", question="qu", should_answer=False),
    ]
    retriever = Scripted(
        {
            "qa": [result(1, "the red fox jumps", "d.txt", 2.0)],
            "qb": [
                result(1, "blue whale", "other.txt", -1.0),
                result(2, "a BLUE whale sings", "d.txt", -2.0),
            ],
            "qu": [result(1, "unrelated", "d.txt", -5.0)],
        }
    )

    report = score_cases(cases, retriever, RetrievalConfidenceGate(min_top_score=0.0))

    assert report.hit_at_1 == 0.5
    assert report.hit_at_5 == 1.0
    assert report.mrr == pytest.approx(0.75)
    assert report.answerable_pass_rate == 0.5
    assert report.unanswerable_abstain_rate == 1.0
    assert is_relevant(result(1, "red fox", "d.txt", 0.0), cases[0])
