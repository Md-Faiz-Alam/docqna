"""How does DocQ&A do on questions it was never tuned on? (held-out test)

Runs the full production pipeline (`build_rag_pipeline`, the same function the
API uses: hybrid search, reranker, retrieval gate, injection filter, LLM,
grounding check) on the held-out SQuAD 2.0 sample built by
`build_heldout_set.py`. No setting was changed for this test.

    python -m experiments.evaluate_heldout run --provider gemini
    python -m experiments.evaluate_heldout run --provider ollama
    python -m experiments.evaluate_heldout analyze

Grading:
- answerable: correct when the shown answer contains one of the crowd-written
  answers, after SQuAD's standard normalisation (lower case, no punctuation,
  no articles). Paraphrased answers ("in 1066" vs "1066") still match; an
  answer that says the same thing in other words does not, so this is strict.
- unanswerable: correct when the system refuses. Any other answer counts as
  made up.

For every refusal the record shows who refused: the retrieval gate (the LLM
was never called), the LLM itself, or a check after writing (grounding,
citations, link check). Results go to `data/evaluation/results/heldout/`.
"""

import argparse
import json
import re
import shutil
import string
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

from docqna.api.dependencies import build_llm, build_rag_pipeline
from docqna.rag import ABSTENTION_ANSWER
from experiments.attack_prompt_injection import RecordingLLM
from experiments.calibrate_answerability import (
    GEMINI_MIN_INTERVAL_SECONDS,
    PROVIDERS,
    SESSION_ID,
    TOP_K,
    _build_retriever,
    experiment_settings,
    with_retries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "heldout"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results" / "heldout"


def normalize(text: str) -> str:
    """SQuAD's answer normalisation: lower case, no punctuation or articles."""

    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) | {"–", "—"})
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_answer(text: str, answers: list[str]) -> bool:
    normalized = f" {normalize(text)} "
    return any(f" {normalize(answer)} " in normalized for answer in answers if normalize(answer))


def load_cases() -> tuple[list[Path], list[dict[str, Any]]]:
    payload = json.loads((HELDOUT_DIR / "cases.json").read_text("utf-8"))
    return sorted((HELDOUT_DIR / "corpus").glob("*.txt")), payload["cases"]


def results_dir(provider: str) -> Path:
    return RESULTS_DIR / ("phi3_mini" if provider == "ollama" else provider)


def refused_by(record: dict[str, Any]) -> str | None:
    """Which part of the pipeline produced a refusal, if any."""

    if record["answered"]:
        return None
    if not record["llm_called"]:
        return "retrieval_gate"
    if record["written"].strip() == ABSTENTION_ANSWER:
        return "llm"
    return "check_after_writing"


def run(provider: str, ollama_url: str) -> Path:
    documents, cases = load_cases()
    out = results_dir(provider) / "cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.read_text("utf-8").splitlines()}

    settings = experiment_settings(provider, ollama_url)
    interval = GEMINI_MIN_INTERVAL_SECONDS if provider == "gemini" else 0.0
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-heldout-"))
    filters: dict[str, str | int] = {"session_id": SESSION_ID}
    try:
        reranker = _build_retriever(documents, settings, work_dir)
        llm = RecordingLLM(build_llm(settings))
        if not llm.health_check():
            raise SystemExit(f"{provider} ({settings.llm_model}) is not reachable.")
        pipeline = build_rag_pipeline(
            settings=settings,
            session_retriever=reranker.base_retriever,  # type: ignore[arg-type]
            cross_encoder_model=reranker.model,
            llm=llm,
        )

        with out.open("a", encoding="utf-8") as sink:
            for number, case in enumerate(cases, start=1):
                if case["case_id"] in done:
                    continue
                answers = case.get("answers", [])
                top = reranker.retrieve(case["question"], top_k=TOP_K, metadata_filters=filters)
                llm.outputs = []
                started = time.monotonic()
                response = with_retries(
                    partial(
                        pipeline.ask_with_citations,
                        case["question"],
                        top_k=TOP_K,
                        metadata_filters=filters,
                    )
                )
                elapsed = time.monotonic() - started
                answered = response.answer != ABSTENTION_ANSWER
                cited = [s for s in response.sources if s.citation_id in response.cited_source_ids]
                record: dict[str, Any] = {
                    "case_id": case["case_id"],
                    "should_answer": case["should_answer"],
                    "question": case["question"],
                    "answers": answers,
                    "top_score": top[0].score if top else None,
                    "evidence_in_top_k": (
                        any(
                            r.filename == case["document"] and contains_answer(r.text, answers)
                            for r in top
                        )
                        if case["should_answer"]
                        else None
                    ),
                    "llm_called": bool(llm.outputs),
                    "written": "\n".join(llm.outputs),
                    "answer": response.answer,
                    "answered": answered,
                    "cited_relevant": answered
                    and any(
                        s.filename == case["document"] and contains_answer(s.text, answers)
                        for s in cited
                    ),
                    "correct": (answered and contains_answer(response.answer, answers))
                    if case["should_answer"]
                    else not answered,
                    "seconds": round(elapsed, 2),
                    "provider": provider,
                    "model": settings.llm_model,
                }
                record["refused_by"] = refused_by(record)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"[{provider} {number}/{len(cases)}] {case['case_id']} "
                    f"should_answer={case['should_answer']} answered={answered} "
                    f"correct={record['correct']}",
                    flush=True,
                )
                if llm.outputs:
                    time.sleep(max(0.0, interval - elapsed))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records if r["should_answer"]]
    unanswerable = [r for r in records if not r["should_answer"]]

    def refusals(rows: list[dict[str, Any]]) -> dict[str, int]:
        counts = {"retrieval_gate": 0, "llm": 0, "check_after_writing": 0}
        for r in rows:
            if r["refused_by"]:
                counts[r["refused_by"]] += 1
        return counts

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "evidence_in_top_k": sum(bool(r["evidence_in_top_k"]) for r in answerable),
        "correct": sum(r["correct"] for r in answerable),
        "wrong": sum(r["answered"] and not r["correct"] for r in answerable),
        "declined_answerable": sum(not r["answered"] for r in answerable),
        "refused_unanswerable": sum(not r["answered"] for r in unanswerable),
        "made_up": sum(r["answered"] for r in unanswerable),
        "refusals_answerable": refusals(answerable),
        "refusals_unanswerable": refusals(unanswerable),
    }


def analyze() -> dict[str, Any]:
    result = {}
    for path in sorted(RESULTS_DIR.glob("*/cases.jsonl")):
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        result[path.parent.name] = {"model": records[0]["model"], **summarize(records)}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    for name, s in result.items():
        print(
            f"{name} ({s['model']}): evidence found {s['evidence_in_top_k']}/{s['answerable']}, "
            f"correct {s['correct']}/{s['answerable']}, wrong {s['wrong']}, "
            f"declined {s['declined_answerable']}, made up {s['made_up']}/{s['unanswerable']}"
        )
        print(f"  refusals of answerable: {s['refusals_answerable']}")
        print(f"  refusals of unanswerable: {s['refusals_unanswerable']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    run_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "run":
        out = run(args.provider, args.ollama_url)
        print(f"wrote {out}")
    else:
        analyze()


if __name__ == "__main__":
    main()
