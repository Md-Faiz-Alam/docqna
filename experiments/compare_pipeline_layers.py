"""What does each layer of the pipeline add? (ablation study)

Starts from plain RAG and adds one layer at a time, on the same 105
questions (75 public, 30 private) as `calibrate_answerability.py`:

1. dense      plain RAG: meaning (embedding) search only, then the LLM answers
2. hybrid     + keyword search (BM25), fused with RRF
3. reranked   + cross-encoder reranker (the production retriever)
4. gated      + retrieval gate: skip clearly off-topic questions
5. grounded   + grounding check after writing (the full system)

The prompt, LLM, temperature and top-k stay the same in every step, so
any change comes from the added layer. Steps 1-2 are answered by the LLM
here. Step 3 reuses the answers already recorded by
`calibrate_answerability.py` (same retriever, temperature 0) and only
re-runs retrieval to measure whether the evidence was found. Steps 4-5 only
decide whether step 3's answer is shown, so they are computed exactly from
those records.

    python -m experiments.compare_pipeline_layers run --dataset public --provider gemini
    python -m experiments.compare_pipeline_layers run --dataset private --provider gemini
    python -m experiments.compare_pipeline_layers analyze --provider gemini

Results go to `data/evaluation/results/pipeline_layers/<provider>/`.
"""

import argparse
import json
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

from docqna.api.dependencies import build_llm
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline
from docqna.retrieval.base import BaseRetriever
from docqna.settings import Settings
from experiments.calibrate_answerability import (
    GEMINI_MIN_INTERVAL_SECONDS,
    PROVIDERS,
    SESSION_ID,
    TOP_K,
    _build_retriever,
    answer_checkers,
    experiment_settings,
    outcomes_at,
    private_cases,
    public_cases,
    results_dir,
    with_retries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "data" / "evaluation" / "results" / "pipeline_layers"
DATASETS = ("public", "private")

LAYERS = {
    "dense": "Plain RAG (meaning search only)",
    "hybrid": "+ keyword search (hybrid)",
    "reranked": "+ reranker",
    "gated": "+ retrieval gate",
    "grounded": "+ grounding check (full system)",
}
LLM_LAYERS = ("dense", "hybrid")
RECORDED_LAYERS = (*LLM_LAYERS, "reranked")


def layer_dir(provider: str) -> Path:
    return RESULTS_ROOT / ("phi3_mini" if provider == "ollama" else provider)


def _retrievers(reranker: Any) -> dict[str, BaseRetriever]:
    """The three retrievers, each one layer more than the last."""

    hybrid = reranker.base_retriever
    return {"dense": hybrid.dense_retriever, "hybrid": hybrid, "reranked": reranker}


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def run(dataset: str, layer: str, provider: str, ollama_url: str) -> Path:
    """Record one layer's outcome for every case (resumable)."""

    documents, cases = public_cases() if dataset == "public" else private_cases()
    out = layer_dir(provider) / f"{dataset}_{layer}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["case_id"] for r in _load(out)}

    recorded: dict[str, dict[str, Any]] = {}
    if layer == "reranked":
        source = results_dir(provider) / f"{dataset}_cases.jsonl"
        recorded = {r["case_id"]: r for r in _load(source)}
        if not recorded:
            raise SystemExit(f"Run calibrate_answerability for {provider}/{dataset} first.")

    settings = experiment_settings(provider, ollama_url)
    interval = GEMINI_MIN_INTERVAL_SECONDS if provider == "gemini" else 0.0
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-layers-"))
    filters: dict[str, str | int] = {"session_id": SESSION_ID}
    try:
        retriever = _retrievers(_build_retriever(documents, settings, work_dir))[layer]
        pipeline = None
        if layer in LLM_LAYERS:
            llm = build_llm(settings)
            if not llm.health_check():
                raise SystemExit(f"{provider} ({settings.llm_model}) is not reachable.")
            pipeline = RAGPipeline(retriever=retriever, llm=llm, confidence_gate=None)

        with out.open("a", encoding="utf-8") as sink:
            for number, case in enumerate(cases, start=1):
                if case.case_id in done:
                    continue
                top = retriever.retrieve(case.question, top_k=TOP_K, metadata_filters=filters)
                sources = RAGPipeline._build_sources(top)
                record: dict[str, Any] = {
                    "case_id": case.case_id,
                    "layer": layer,
                    "should_answer": case.should_answer,
                    "evidence_in_top_k": (
                        any(case.is_relevant(s) for s in sources) if case.should_answer else None
                    ),
                }
                if pipeline is not None:
                    started = time.monotonic()
                    response = with_retries(
                        partial(
                            pipeline.ask_with_citations,
                            case.question,
                            top_k=TOP_K,
                            metadata_filters=filters,
                        )
                    )
                    elapsed = time.monotonic() - started
                    answered = response.answer != ABSTENTION_ANSWER
                    cited = [
                        s for s in response.sources if s.citation_id in response.cited_source_ids
                    ]
                    record.update(
                        answered=answered,
                        cited_relevant=answered and any(case.is_relevant(s) for s in cited),
                        answer=response.answer,
                        seconds=round(elapsed, 2),
                    )
                else:
                    prior = recorded[case.case_id]
                    record.update(
                        {
                            k: prior[k]
                            for k in (
                                "answered",
                                "cited_relevant",
                                "answer",
                                "top_score",
                                "grounding_score",
                                "seconds",
                            )
                            if k in prior
                        }
                    )
                record.update(provider=provider, model=settings.llm_model)
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"[{provider} {dataset} {layer} {number}/{len(cases)}] {case.case_id} "
                    f"answered={record['answered']} cited_relevant={record['cited_relevant']}",
                    flush=True,
                )
                if pipeline is not None:
                    time.sleep(max(0.0, interval - record["seconds"]))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return out


def summarize(
    records: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    grounding_min: float | None = None,
) -> dict[str, Any]:
    """Counts for one layer. `correct` must already be set on each record.

    Without a threshold every recorded answer is shown. With one, the gate
    (and optionally the grounding check) decides which answers are shown,
    exactly as in production.
    """

    answerable = [r for r in records if r["should_answer"]]
    unanswerable = [r for r in records if not r["should_answer"]]

    if threshold is None:
        correct = sum(bool(r.get("correct")) for r in answerable)
        wrong = sum(bool(r["answered"]) and not r.get("correct") for r in answerable)
        false_answers = sum(bool(r["answered"]) for r in unanswerable)
    else:
        shown = outcomes_at(records, threshold, grounding_min)
        correct = round(shown["answerable_correct"] * len(answerable))
        wrong = round(shown["answerable_wrong_answer"] * len(answerable))
        false_answers = round(shown["unanswerable_false_answer"] * len(unanswerable))

    evidence = [r for r in answerable if r.get("evidence_in_top_k") is not None]
    seconds = sorted(r["seconds"] for r in records if r.get("seconds") is not None)
    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "correct": correct,
        "wrong": wrong,
        "declined_answerable": len(answerable) - correct - wrong,
        "false_answers": false_answers,
        "evidence_in_top_k": sum(bool(r["evidence_in_top_k"]) for r in evidence)
        if evidence
        else None,
        "median_seconds": seconds[len(seconds) // 2] if seconds else None,
    }


def analyze(provider: str) -> dict[str, Any]:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    threshold = settings.answerability_min_score
    grounding = (
        settings.gemini_grounding_min_score
        if provider == "gemini"
        else settings.grounding_min_score
    )
    checkers = answer_checkers()
    out_dir = layer_dir(provider)

    table: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        by_layer: dict[str, list[dict[str, Any]]] = {}
        for layer in RECORDED_LAYERS:
            records = _load(out_dir / f"{dataset}_{layer}.jsonl")
            for r in records:
                check = checkers[dataset].get(r["case_id"])
                r["correct"] = bool(r["answered"]) and (
                    bool(r["cited_relevant"]) or bool(check and check(r["answer"]))
                )
            by_layer[layer] = records
        if not all(by_layer.values()):
            missing = [layer for layer, rows in by_layer.items() if not rows]
            print(f"skipping {dataset}: no records for {missing}")
            continue
        table[dataset] = {
            "dense": summarize(by_layer["dense"]),
            "hybrid": summarize(by_layer["hybrid"]),
            "reranked": summarize(by_layer["reranked"]),
            "gated": summarize(by_layer["reranked"], threshold=threshold),
            "grounded": summarize(
                by_layer["reranked"], threshold=threshold, grounding_min=grounding
            ),
        }
        for layer in ("gated", "grounded"):
            table[dataset][layer]["evidence_in_top_k"] = table[dataset]["reranked"][
                "evidence_in_top_k"
            ]

    combined = (
        {
            layer: {
                key: sum(table[d][layer][key] or 0 for d in table)
                for key in (
                    "answerable",
                    "unanswerable",
                    "correct",
                    "wrong",
                    "declined_answerable",
                    "false_answers",
                    "evidence_in_top_k",
                )
            }
            for layer in LAYERS
        }
        if table
        else {}
    )

    result = {
        "provider": provider,
        "model": settings.gemini_model if provider == "gemini" else settings.ollama_model,
        "retrieval_gate": threshold,
        "grounding_min": grounding,
        "top_k": TOP_K,
        "layers": LAYERS,
        "by_dataset": table,
        "combined": combined,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(render_table(result))
    return result


def render_table(result: dict[str, Any]) -> str:
    lines = [
        f"\n{result['provider']} ({result['model']})",
        f"{'layer':<34}{'evidence found':>15}{'correct':>10}{'wrong':>8}{'made-up':>10}",
    ]
    for layer, label in LAYERS.items():
        row = result["combined"].get(layer)
        if not row:
            continue
        lines.append(
            f"{label:<34}{row['evidence_in_top_k']:>8}/{row['answerable']:<6}"
            f"{row['correct']:>4}/{row['answerable']:<5}{row['wrong']:>6}"
            f"{row['false_answers']:>6}/{row['unanswerable']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--layer", choices=RECORDED_LAYERS, action="append")
    run_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    analyze_parser = sub.add_parser("analyze")
    for command_parser in (run_parser, analyze_parser):
        command_parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    args = parser.parse_args()

    if args.command == "run":
        for layer in args.layer or RECORDED_LAYERS:
            print(f"wrote {run(args.dataset, layer, args.provider, args.ollama_url)}")
    else:
        analyze(args.provider)


if __name__ == "__main__":
    main()
