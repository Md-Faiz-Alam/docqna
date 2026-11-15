"""Does the automatic grading agree with a human? (grading check)

Every result in `docs/evaluation.md` is graded by rules, not by people. An
answer counts as correct when it states the labelled fact, cites a passage
holding the evidence (public set) or contains a crowd-written answer (SQuAD).
Any answer to an unanswerable question counts as made up. This script checks
those rules against a person's judgement.

    python -m experiments.grading_check sample    # draw the 80 answers (once)
    python -m experiments.grading_check label     # labelling page on 127.0.0.1
    python -m experiments.grading_check analyze   # compare labels with the rules

The sample holds only answers the final system actually showed a user (refusals
need no judgement), drawn with a fixed seed: 20 each from Gemini and
`phi3:mini` on the public test set and on the unseen SQuAD 2.0 set. Only public
data is involved. The labelling page hides the model and the automatic grade,
and shows the answers in a shuffled order.

Labels: correct, partly correct, wrong (or misleading, or unsupported), or
can't tell. The automatic grade is compared with the human one twice: strict
(partly correct counts as wrong) and lenient (partly correct counts as
correct). "Can't tell" is left out.
"""

import argparse
import json
import random
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from docqna.evaluation.public_benchmark import load_cases as load_public_cases
from docqna.settings import Settings
from experiments.calibrate_answerability import answer_checkers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "data" / "evaluation" / "results"
CHECK_DIR = RESULTS / "grading_check"
SAMPLE_FILE = CHECK_DIR / "sample.json"
LABELS_FILE = CHECK_DIR / "labels.json"
PAGE_FILE = Path(__file__).resolve().parent / "grading_check_page.html"
PUBLIC_CORPUS = PROJECT_ROOT / "data" / "evaluation" / "public" / "corpus"
HELDOUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "heldout"
SQUAD_RAW = HELDOUT_DIR / "raw" / "dev-v2.0.json"

SEED = 2026
PER_GROUP = 20
PROVIDERS = {"gemini": "gemini", "phi3_mini": "ollama"}
LABELS = ("correct", "partly", "wrong", "unsure")
PAGE_FIELDS = (
    "item_id",
    "question",
    "answer",
    "reference",
    "reference_kind",
    "context",
    "document",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def public_shown(provider_dir: str, settings: Settings) -> list[dict[str, Any]]:
    """Public-set answers the final system shows (gate and grounding check passed)."""

    grounding_min = (
        settings.gemini_grounding_min_score
        if provider_dir == "gemini"
        else settings.grounding_min_score
    )
    check = answer_checkers()["public"]
    cases = {c.case_id: c for c in load_public_cases()}
    rows = []
    for r in _jsonl(RESULTS / "pipeline_layers" / provider_dir / "public_reranked.jsonl"):
        score = r.get("grounding_score")
        shown = (
            r["answered"]
            and r["top_score"] is not None
            and r["top_score"] >= settings.answerability_min_score
            and score is not None
            and score >= grounding_min
        )
        if not shown:
            continue
        case = cases[r["case_id"]]
        if r["should_answer"]:
            correct = bool(r["cited_relevant"]) or bool(check[r["case_id"]](r["answer"]))
            text = (PUBLIC_CORPUS / case.document).read_text("utf-8")
            evidence = [p.lower() for p in case.evidence]
            context = [p for p in _paragraphs(text) if any(e in p.lower() for e in evidence)]
            reference, kind, document = list(case.evidence), "expected", case.document
        else:
            correct, context, reference, kind, document = False, [], [], "unanswerable", None
        rows.append(
            {
                "group": f"public/{provider_dir}",
                "case_id": r["case_id"],
                "should_answer": r["should_answer"],
                "question": case.question,
                "answer": r["answer"],
                "reference": reference,
                "reference_kind": kind,
                "context": "\n\n".join(context),
                "document": document,
                "auto_correct": correct,
            }
        )
    return rows


def heldout_shown(provider_dir: str, squad_context: dict[str, str]) -> list[dict[str, Any]]:
    """Unseen SQuAD answers the final system showed."""

    cases = {
        c["case_id"]: c
        for c in json.loads((HELDOUT_DIR / "cases.json").read_text("utf-8"))["cases"]
    }
    rows = []
    for r in _jsonl(RESULTS / "heldout" / provider_dir / "cases.jsonl"):
        if not r["answered"]:
            continue
        rows.append(
            {
                "group": f"heldout/{provider_dir}",
                "case_id": r["case_id"],
                "should_answer": r["should_answer"],
                "question": r["question"],
                "answer": r["answer"],
                "reference": r["answers"],
                "reference_kind": "expected" if r["should_answer"] else "squad_unanswerable",
                "context": squad_context[r["case_id"]],
                "document": cases[r["case_id"]]["document"],
                "auto_correct": bool(r["correct"]),
            }
        )
    return rows


def squad_contexts() -> dict[str, str]:
    """The paragraph each SQuAD question was written from, by case id."""

    if not SQUAD_RAW.exists():
        raise SystemExit(f"Needs SQuAD 2.0 dev-v2.0.json in {SQUAD_RAW.parent} (see README).")
    data = json.loads(SQUAD_RAW.read_text("utf-8"))["data"]
    return {
        f"squad_{q['id']}": p["context"]
        for article in data
        for p in article["paragraphs"]
        for q in p["qas"]
    }


def draw(groups: dict[str, list[dict[str, Any]]], seed: int = SEED) -> list[dict[str, Any]]:
    """PER_GROUP answers from each group, then one shuffled, numbered list."""

    rng = random.Random(seed)
    items = []
    for name in sorted(groups):
        rows = sorted(groups[name], key=lambda r: r["case_id"])
        items.extend(rng.sample(rows, min(PER_GROUP, len(rows))))
    rng.shuffle(items)
    for number, item in enumerate(items, start=1):
        item["item_id"] = f"g{number:03d}"
    return items


def sample() -> Path:
    if SAMPLE_FILE.exists():
        raise SystemExit(f"{SAMPLE_FILE} exists; delete it first to draw a new sample.")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    contexts = squad_contexts()
    groups = {}
    for provider_dir in PROVIDERS:
        groups[f"public/{provider_dir}"] = public_shown(provider_dir, settings)
        groups[f"heldout/{provider_dir}"] = heldout_shown(provider_dir, contexts)
    items = draw(groups)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Blind sample for the grading check: answers the final system showed, "
            f"{PER_GROUP} per group, drawn by experiments/grading_check.py (seed {SEED}). "
            "SQuAD-derived text is CC BY-SA 4.0 (Rajpurkar, Jia and Liang, 2018)."
        ),
        "available": {name: len(rows) for name, rows in groups.items()},
        "items": items,
    }
    SAMPLE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )
    print(f"wrote {len(items)} answers to {SAMPLE_FILE}")
    return SAMPLE_FILE


def load_items() -> list[dict[str, Any]]:
    return json.loads(SAMPLE_FILE.read_text("utf-8"))["items"]


def load_labels() -> dict[str, dict[str, str]]:
    if not LABELS_FILE.exists():
        return {}
    return json.loads(LABELS_FILE.read_text("utf-8"))["labels"]


def save_labels(labels: dict[str, dict[str, str]]) -> None:
    payload = {"labels": dict(sorted(labels.items()))}
    LABELS_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )


def page_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the labelling page may see: no model, no grade, no case id."""

    return [{key: item[key] for key in PAGE_FIELDS} for item in items]


def document_path(name: str, items: list[dict[str, Any]]) -> Path | None:
    if name not in {item["document"] for item in items if item["document"]}:
        return None
    for folder in (PUBLIC_CORPUS, HELDOUT_DIR / "corpus"):
        if (folder / name).exists():
            return folder / name
    return None


def make_handler(items: list[dict[str, Any]]) -> type[BaseHTTPRequestHandler]:
    visible = page_items(items)
    ids = {item["item_id"] for item in items}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value: Any, status: int = 200) -> None:
            self._send(json.dumps(value).encode("utf-8"), "application/json", status)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(PAGE_FILE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/items":
                self._json({"items": visible, "labels": load_labels()})
            elif url.path == "/api/document":
                name = parse_qs(url.query).get("name", [""])[0]
                path = document_path(name, items)
                if path is None:
                    self._json({"error": "unknown document"}, 404)
                else:
                    self._json({"name": name, "text": path.read_text("utf-8")})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/label":
                self._json({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(min(length, 100_000)) or b"{}")
            item_id, label = body.get("item_id"), body.get("label")
            if item_id not in ids or label not in LABELS:
                self._json({"error": "bad label"}, 400)
                return
            labels = load_labels()
            labels[item_id] = {"label": label, "note": str(body.get("note", ""))[:1000]}
            save_labels(labels)
            self._json({"saved": len(labels)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def label(port: int, open_browser: bool) -> None:
    items = load_items()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(items))
    url = f"http://127.0.0.1:{port}/"
    print(f"Labelling page: {url} ({len(load_labels())}/{len(items)} labelled). Ctrl+C stops.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two yes/no raters."""

    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    p_auto = sum(a for a, _ in pairs) / n
    p_human = sum(b for _, b in pairs) / n
    expected = p_auto * p_human + (1 - p_auto) * (1 - p_human)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 3)


def compare(items: list[dict[str, Any]], labels: dict[str, dict[str, str]], lenient: bool) -> dict:
    """Agreement between the automatic grade and the human label."""

    accepted = {"correct", "partly"} if lenient else {"correct"}
    pairs = []
    for item in items:
        human = labels.get(item["item_id"], {}).get("label")
        if human in (None, "unsure"):
            continue
        pairs.append((bool(item["auto_correct"]), human in accepted))
    both_ok = sum(a and h for a, h in pairs)
    auto_only = sum(a and not h for a, h in pairs)
    human_only = sum(h and not a for a, h in pairs)
    both_bad = sum(not a and not h for a, h in pairs)
    return {
        "compared": len(pairs),
        "agree": both_ok + both_bad,
        "agreement": round((both_ok + both_bad) / len(pairs), 3) if pairs else None,
        "kappa": kappa(pairs),
        "both_correct": both_ok,
        "auto_correct_human_not": auto_only,
        "human_correct_auto_not": human_only,
        "both_not_correct": both_bad,
        "auto_correct_rate": round((both_ok + auto_only) / len(pairs), 3) if pairs else None,
        "human_correct_rate": round((both_ok + human_only) / len(pairs), 3) if pairs else None,
    }


def analyze() -> dict[str, Any]:
    items = load_items()
    labels = load_labels()
    counts = dict.fromkeys(LABELS, 0)
    for entry in labels.values():
        counts[entry["label"]] += 1

    def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "strict": compare(rows, labels, lenient=False),
            "lenient": compare(rows, labels, lenient=True),
        }

    groups = sorted({item["group"] for item in items})
    kinds = {
        "answerable": [i for i in items if i["should_answer"]],
        "unanswerable": [i for i in items if not i["should_answer"]],
    }
    disagreements = []
    for item in items:
        human = labels.get(item["item_id"], {}).get("label")
        if human in (None, "unsure"):
            continue
        if bool(item["auto_correct"]) != (human == "correct"):
            disagreements.append(
                {
                    "item_id": item["item_id"],
                    "group": item["group"],
                    "case_id": item["case_id"],
                    "should_answer": item["should_answer"],
                    "auto_correct": item["auto_correct"],
                    "human": human,
                    "note": labels[item["item_id"]].get("note", ""),
                    "question": item["question"],
                    "answer": item["answer"],
                }
            )
    result: dict[str, Any] = {
        "labelled": len(labels),
        "items": len(items),
        "label_counts": counts,
        "overall": report(items),
        "by_group": {g: report([i for i in items if i["group"] == g]) for g in groups},
        "by_question_type": {k: report(rows) for k, rows in kinds.items()},
        "disagreements": disagreements,
    }
    (CHECK_DIR / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8", newline="\n"
    )
    strict, lenient = result["overall"]["strict"], result["overall"]["lenient"]
    print(f"labelled {len(labels)}/{len(items)}: {counts}")
    print(f"strict:  agreement {strict['agreement']}, kappa {strict['kappa']}")
    print(f"lenient: agreement {lenient['agreement']}, kappa {lenient['kappa']}")
    for group, rows in result["by_group"].items():
        print(f"  {group}: strict {rows['strict']['agree']}/{rows['strict']['compared']}")
    print(f"{len(disagreements)} disagreements (strict) in summary.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample")
    label_parser = sub.add_parser("label")
    label_parser.add_argument("--port", type=int, default=8765)
    label_parser.add_argument("--no-browser", action="store_true")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "sample":
        sample()
    elif args.command == "label":
        label(args.port, not args.no_browser)
    else:
        analyze()


if __name__ == "__main__":
    main()
