"""Export the measured evaluation results the website shows to web/src/data/results.json.

The website's results page and landing-page numbers come only from this
file, so every number on the site traces back to a committed experiment.
Re-run after any evaluation run:

    python -m scripts.export_web_results

The answer-quality numbers need the per-case LLM results, including the
git-ignored private ones, so run this where the experiments were run.
"""

import json
from pathlib import Path
from typing import Any

from docqna.settings import Settings
from experiments.calibrate_answerability import answer_checkers, outcomes_at, results_dir

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "evaluation" / "results"
OUT = ROOT / "web" / "src" / "data" / "results.json"

RETRIEVER_LABELS = {
    "dense": "Dense (embeddings)",
    "bm25": "BM25 (keywords)",
    "rrf": "Hybrid (RRF)",
    "rrf_reranker": "Hybrid + reranker",
}
CHUNKING_LABELS = {
    "fixed": "Fixed size",
    "structure_aware": "Structure-aware",
    "semantic": "Semantic",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def answer_quality(provider: str, label: str, settings: Settings) -> dict[str, Any]:
    checkers = answer_checkers()
    grounding = (
        settings.gemini_grounding_min_score
        if provider == "gemini"
        else settings.grounding_min_score
    )
    row: dict[str, Any] = {"provider": provider, "label": label, "grounding_min": grounding}
    for dataset in ("public", "private"):
        records = [
            json.loads(line)
            for line in (results_dir(provider) / f"{dataset}_cases.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        for r in records:
            check = checkers[dataset].get(r["case_id"])
            r["correct"] = r["answered"] and (
                r["cited_relevant"] or bool(check and check(r["answer"]))
            )
        outcome = outcomes_at(records, settings.answerability_min_score, grounding)
        row[dataset] = {
            "cases": len(records),
            "answerable": sum(r["should_answer"] for r in records),
            "unanswerable": sum(not r["should_answer"] for r in records),
            "correct": _r(outcome["answerable_correct"]),
            "wrong": _r(outcome["answerable_wrong_answer"]),
            "false_answers": _r(outcome["unanswerable_false_answer"]),
        }
        seconds = [r["seconds"] for r in records if "seconds" in r]
        if seconds:
            row[dataset]["median_seconds"] = round(sorted(seconds)[len(seconds) // 2], 2)
    return row


def pipeline_layers(provider: str, label: str) -> dict[str, Any]:
    """Combined counts per layer from experiments/compare_pipeline_layers.py."""

    folder = "phi3_mini" if provider == "ollama" else provider
    summary = _load(RESULTS / "pipeline_layers" / folder / "summary.json")
    return {
        "provider": provider,
        "label": label,
        "layers": [
            {
                "key": key,
                "label": name,
                **{
                    field: summary["combined"][key][field]
                    for field in (
                        "answerable",
                        "unanswerable",
                        "evidence_in_top_k",
                        "correct",
                        "wrong",
                        "false_answers",
                    )
                },
            }
            for key, name in summary["layers"].items()
        ],
    }


def prompt_injection(provider: str, label: str) -> dict[str, Any] | None:
    """Attack results before and after the defence, from attack_prompt_injection.py."""

    folder = "phi3_mini" if provider == "ollama" else provider
    path = RESULTS / "prompt_injection" / folder / "summary.json"
    if not path.exists():
        return None
    summary = _load(path)
    runs = summary["runs"]
    if not {"baseline", "defended"} <= runs.keys():
        return None
    return {
        "provider": provider,
        "label": label,
        "attacks": [
            {
                "key": key,
                "label": name,
                "cases": runs["baseline"]["by_attack"][key]["cases"],
                "before": runs["baseline"]["by_attack"][key]["hijacked_shown"],
                "after": runs["defended"]["by_attack"][key]["hijacked_shown"],
            }
            for key, name in summary["attack_labels"].items()
            if key in runs["baseline"]["by_attack"]
        ],
        "overall": {
            stage: {
                "cases": runs[stage]["overall"]["cases"],
                "llm_fell_for_it": runs[stage]["overall"]["hijacked_written"],
                "user_saw_it": runs[stage]["overall"]["hijacked_shown"],
                "real_answer_cases": runs[stage]["overall"]["real_answer_cases"],
                "real_answer_correct": runs[stage]["overall"]["real_answer_correct"],
            }
            for stage in ("baseline", "defended")
        },
    }


def speed_and_cost(settings: Settings) -> list[dict[str, Any]]:
    """Per-stage timings, tokens and cost from measure_speed_and_cost.py."""

    path = RESULTS / "speed_and_cost" / "summary.json"
    if not path.exists():
        return []
    labels = {"phi3_mini": "phi3:mini (local)", "gemini": f"Gemini {settings.gemini_model}"}
    rows = []
    for folder, s in _load(path).items():
        public_docs = [d for d in s["indexing"] if not d["document"].startswith("private")]
        private_docs = [d for d in s["indexing"] if d["document"].startswith("private")]
        rows.append(
            {
                "provider": s["provider"],
                "label": labels.get(folder, folder),
                "processor": s["hardware"]["processor"],
                "logical_cpus": s["hardware"]["logical_cpus"],
                "questions": s["questions"],
                "stages_ms": {
                    stage: {"median": v["median"], "p95": v["p95"]}
                    for stage, v in s["stages_ms"].items()
                },
                "tokens": s["mean_tokens_per_question"],
                "cost_per_1000_usd": (s["cost_usd"] or {}).get("per_1000_questions"),
                "startup_seconds": round(sum(s["startup_seconds"].values()), 1),
                "index_public_seconds": round(sum(d["seconds"] for d in public_docs), 2),
                "index_private": private_docs[0] if private_docs else None,
            }
        )
    return rows


def heldout(settings: Settings) -> dict[str, Any] | None:
    """Unseen SQuAD 2.0 results from evaluate_heldout.py, with Gemini's hand review."""

    path = RESULTS / "heldout" / "summary.json"
    if not path.exists():
        return None
    labels = {"phi3_mini": "phi3:mini (local)", "gemini": f"Gemini {settings.gemini_model}"}
    rows = [
        {
            "provider": folder,
            "label": labels.get(folder, folder),
            **{
                key: s[key]
                for key in (
                    "answerable",
                    "unanswerable",
                    "evidence_in_top_k",
                    "correct",
                    "wrong",
                    "declined_answerable",
                    "made_up",
                )
            },
        }
        for folder, s in _load(path).items()
    ]
    review_path = RESULTS / "heldout" / "gemini" / "manual_review.json"
    review = None
    if review_path.exists():
        r = _load(review_path)
        review = {
            key: len(group["cases"]) for key, group in r["unanswerable_but_answered"].items()
        } | {key: len(cases) for key, cases in r["answerable_not_correct"].items()}
    return {"rows": rows, "gemini_review": review}


def grading_check() -> dict[str, Any] | None:
    """Agreement between the automatic grading and the AI-assisted human labels."""

    path = RESULTS / "grading_check" / "summary.json"
    if not path.exists():
        return None
    s = _load(path)

    def row(key: str, label: str, report: dict[str, Any]) -> dict[str, Any]:
        strict = report["strict"]
        return {
            "key": key,
            "label": label,
            "compared": strict["compared"],
            "agree": strict["agree"],
            "kappa": strict["kappa"],
            "auto_too_harsh": strict["human_correct_auto_not"],
            "auto_too_generous": strict["auto_correct_human_not"],
        }

    kinds = s["by_question_type"]
    return {
        "labelled": s["labelled"],
        "label_counts": s["label_counts"],
        "rows": [
            row("answerable", "Normal questions", kinds["answerable"]),
            row("unanswerable", "SQuAD trick questions", kinds["unanswerable"]),
            row("overall", "All answers", s["overall"]),
        ],
    }


def main() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    retrieval = [
        {
            "key": r["retriever"],
            "label": RETRIEVER_LABELS.get(r["retriever"], r["retriever"]),
            "hit_at_1": _r(r["hit_at_1"]),
            "hit_at_5": _r(r["hit_at_5"]),
            "mrr": _r(r["mrr_at_5"]),
        }
        for r in _load(RESULTS / "retrieval_methods" / "retrieval_comparison.json")["results"]
    ]
    chunking = [
        {
            "key": r["strategy"],
            "label": CHUNKING_LABELS.get(r["strategy"], r["strategy"]),
            "hit_at_1": _r(r["hit_at_1"]),
            "mrr": _r(r["mrr_at_5"]),
        }
        for r in _load(RESULTS / "chunking_strategies" / "chunking_comparison.json")["results"]
    ]
    benchmark = _load(RESULTS / "public_benchmark.json")
    sweep = _load(RESULTS / "answerability/phi3_mini" / "threshold_sweep.json")
    gate_only = [
        {
            "threshold": row["threshold"],
            "correct": _r(row["answerable_correct"]),
            "false_answers": _r(row["unanswerable_false_answer"]),
        }
        for row in sweep["gate_only"]["public"]
        if row["threshold"] not in ("-inf", float("-inf"))
    ]

    data = {
        "source": "Generated by scripts/export_web_results.py from data/evaluation/results/",
        "public_benchmark": {
            "answerable": benchmark["num_answerable"],
            "unanswerable": benchmark["num_unanswerable"],
            "hit_at_1": _r(benchmark["hit_at_1"]),
            "hit_at_3": _r(benchmark["hit_at_3"]),
            "hit_at_5": _r(benchmark["hit_at_5"]),
            "mrr": _r(benchmark["mrr"]),
        },
        "retrieval": retrieval,
        "chunking": chunking,
        "answer_quality": [
            answer_quality("ollama", "phi3:mini (local)", settings),
            answer_quality("gemini", f"Gemini {settings.gemini_model}", settings),
        ],
        "pipeline_layers": [
            pipeline_layers("ollama", "phi3:mini (local)"),
            pipeline_layers("gemini", f"Gemini {settings.gemini_model}"),
        ],
        "prompt_injection": [
            row
            for row in (
                prompt_injection("ollama", "phi3:mini (local)"),
                prompt_injection("gemini", f"Gemini {settings.gemini_model}"),
            )
            if row is not None
        ],
        "speed_and_cost": speed_and_cost(settings),
        "heldout": heldout(settings),
        "grading_check": grading_check(),
        "old_gate": {
            "label": "Before recalibration (gate >= 0)",
            "public_correct": _r(sweep["selection"]["old"]["public"]["answerable_correct"]),
            "public_false_answers": _r(
                sweep["selection"]["old"]["public"]["unanswerable_false_answer"]
            ),
        },
        "gate_only_sweep_phi3_public": gate_only,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
