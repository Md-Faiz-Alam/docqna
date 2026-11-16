"""Whole-answer vs per-sentence grounding check, re-scored offline.

The production grounding check scores the whole answer, as one claim, against
every span of every cited source, and keeps the best match. Two weaknesses:
an answer with one supported sentence and one made-up sentence can pass on
the strength of the supported one, and a long correct answer that combines
several facts matches no single sentence well and is withheld.

The per-sentence check (`GroundingVerifier.evaluate_claims`) scores each
sentence against the sources *that sentence* cites, and the answer's score is
its weakest sentence.

No LLM is called. The recorded LLM answers (written before any check) are
re-scored, because retrieval is deterministic: rebuilding the index
reproduces the passages each answer's citation numbers refer to.

    python -m experiments.compare_grounding_checks

Datasets: public (answerability records, both providers) and held-out SQuAD
2.0 (both providers). The private set needs a document that is not in the
repository, so it is skipped. Results go to
`data/evaluation/results/grounding_checks/`.
"""

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docqna.rag import ABSTENTION_ANSWER
from docqna.rag.grounding import GroundingVerifier
from docqna.rag.injection import screen_passages
from docqna.rag.pipeline import RAGPipeline
from docqna.settings import Settings
from experiments.calibrate_answerability import (
    SESSION_ID,
    TOP_K,
    _build_retriever,
    answer_checkers,
    public_cases,
)
from experiments.evaluate_heldout import contains_answer, load_cases

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "data" / "evaluation" / "results"
OUT_DIR = RESULTS / "grounding_checks"
PROVIDERS = {"phi3_mini": "ollama", "gemini": "gemini"}
THRESHOLDS = [x / 2 for x in range(-8, 21)]
PRINTED = {0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0}


def _public_records(folder: str) -> list[dict[str, Any]]:
    path = RESULTS / "answerability" / folder / "public_cases.jsonl"
    check = answer_checkers()["public"]
    records = []
    for line in path.read_text("utf-8").splitlines():
        r = json.loads(line)
        checker = check.get(r["case_id"])
        records.append(
            {
                "case_id": r["case_id"],
                "should_answer": r["should_answer"],
                "top_score": r["top_score"],
                "answer": r["answer"] if r["answered"] else ABSTENTION_ANSWER,
                "correct": bool(
                    r["answered"]
                    and (r["cited_relevant"] or (checker is not None and checker(r["answer"])))
                ),
            }
        )
    return records


def _clean(raw: str) -> str:
    return RAGPipeline._normalize_citations(RAGPipeline._sanitize_generated_answer(raw.strip()))


def _heldout_records(folder: str) -> list[dict[str, Any]]:
    path = RESULTS / "heldout" / folder / "cases.jsonl"
    records = []
    for line in path.read_text("utf-8").splitlines():
        r = json.loads(line)
        # The checks run on the cleaned answer (labels stripped, "[1, 5]" -> "[1][5]").
        written = _clean(r["written"]) if r["llm_called"] else ABSTENTION_ANSWER
        records.append(
            {
                "case_id": r["case_id"],
                "question": r["question"],
                "should_answer": r["should_answer"],
                "top_score": r["top_score"],
                "answer": written,
                "correct": bool(
                    r["should_answer"]
                    and written.strip() != ABSTENTION_ANSWER
                    and contains_answer(written, r["answers"])
                ),
            }
        )
    return records


def _score(
    records: list[dict[str, Any]], documents: list[Path], questions: dict[str, str]
) -> list[dict[str, Any]]:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-grounding-"))
    try:
        reranker = _build_retriever(documents, settings, work_dir)
        verifier = GroundingVerifier(scorer=reranker.model, min_score=0.0)
        filters: dict[str, str | int] = {"session_id": SESSION_ID}
        for r in records:
            r["whole_score"] = r["sentence_score"] = None
            if r["answer"].strip() == ABSTENTION_ANSWER:
                continue
            results, _ = screen_passages(
                reranker.retrieve(questions[r["case_id"]], top_k=TOP_K, metadata_filters=filters)
            )
            texts = [result.text for result in results]
            cited = RAGPipeline._extract_citation_ids(r["answer"])
            cited_texts = [texts[i - 1] for i in cited if 1 <= i <= len(texts)]
            r["whole_score"] = verifier.evaluate(r["answer"], cited_texts).score
            r["sentence_score"] = verifier.evaluate_claims(r["answer"], texts).score
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return records


def outcomes(
    records: list[dict[str, Any]], key: str, threshold: float, gate: float
) -> dict[str, Any]:
    """What the user would see with the retrieval gate and this grounding check."""

    def shown(r: dict[str, Any]) -> bool:
        if r["top_score"] is None or r["top_score"] < gate:
            return False
        score = r[key]
        return score is not None and score >= threshold

    answerable = [r for r in records if r["should_answer"]]
    unanswerable = [r for r in records if not r["should_answer"]]
    return {
        "threshold": threshold,
        "correct": sum(shown(r) and r["correct"] for r in answerable),
        "wrong": sum(shown(r) and not r["correct"] for r in answerable),
        "made_up": sum(shown(r) for r in unanswerable),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
    }


def run() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    gate = settings.answerability_min_score
    current = {
        "phi3_mini": settings.grounding_min_score,
        "gemini": settings.gemini_grounding_min_score,
    }

    public_docs, cases = public_cases()
    public_questions = {case.case_id: case.question for case in cases}
    heldout_docs, heldout_cases = load_cases()
    heldout_questions = {case["case_id"]: case["question"] for case in heldout_cases}

    loaders: dict[str, tuple[Callable[[str], list[dict[str, Any]]], list[Path], dict[str, str]]] = {
        "public": (_public_records, public_docs, public_questions),
        "heldout": (_heldout_records, heldout_docs, heldout_questions),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"retrieval_gate": gate, "current_thresholds": current, "runs": {}}
    for dataset, (load, documents, questions) in loaders.items():
        for folder in PROVIDERS:
            records = _score(load(folder), documents, questions)
            (OUT_DIR / f"{dataset}_{folder}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", "utf-8"
            )
            summary["runs"][f"{dataset}/{folder}"] = {
                "whole_at_current": outcomes(records, "whole_score", current[folder], gate),
                "sentence_sweep": [
                    outcomes(records, "sentence_score", t, gate) for t in THRESHOLDS
                ],
            }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    for name, run_ in summary["runs"].items():
        w = run_["whole_at_current"]
        print(
            f"\n{name}: whole-answer @ {w['threshold']}: correct {w['correct']}/{w['answerable']} "
            f"wrong {w['wrong']} made-up {w['made_up']}/{w['unanswerable']}"
        )
        for row in (r for r in run_["sentence_sweep"] if r["threshold"] in PRINTED):
            print(
                f"  per-sentence @ {row['threshold']:>5}: correct {row['correct']:>3} "
                f"wrong {row['wrong']:>3} made-up {row['made_up']:>3}"
            )


if __name__ == "__main__":
    run()
