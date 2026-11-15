"""Build the held-out test set from SQuAD 2.0 (never used for tuning).

Every threshold in DocQ&A was chosen on the public and private test sets, so
results on those sets are optimistic. This script samples an unseen set from
the SQuAD 2.0 development set: Wikipedia articles with crowd-written
questions, about half of them unanswerable. The unanswerable ones were written
to look answerable from the same paragraph, which makes them a hard test of
refusing to make things up.

The articles and questions are drawn with a fixed seed, chosen before any
result was seen, so the sample is reproducible and not picked by hand:

    python -m experiments.build_heldout_set

Input: `data/evaluation/heldout/raw/dev-v2.0.json`, downloaded from
https://rajpurkar.github.io/SQuAD-explorer/ (git-ignored).
Output: one text document per article in `data/evaluation/heldout/corpus/`
and `data/evaluation/heldout/cases.json`.

SQuAD 2.0 is licensed CC BY-SA 4.0 (Rajpurkar, Jia and Liang, 2018); the
derived files carry the same licence.
"""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "heldout"
RAW_FILE = HELDOUT_DIR / "raw" / "dev-v2.0.json"
RAW_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"

SEED = 2026
ARTICLES = 5
ANSWERABLE_PER_ARTICLE = 10
UNANSWERABLE_PER_ARTICLE = 10


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def article_text(article: dict[str, Any]) -> str:
    title = article["title"].replace("_", " ")
    paragraphs = [p["context"].strip() for p in article["paragraphs"]]
    return title + "\n\n" + "\n\n".join(paragraphs) + "\n"


def sample(data: list[dict[str, Any]], seed: int = SEED) -> tuple[list[dict[str, Any]], list[dict]]:
    """Pick articles, then answerable and unanswerable questions in each."""

    rng = random.Random(seed)
    articles = rng.sample(sorted(data, key=lambda a: a["title"]), ARTICLES)
    cases = []
    for article in articles:
        document = f"{slug(article['title'])}.txt"
        questions = [q for p in article["paragraphs"] for q in p["qas"]]
        answerable = [q for q in questions if not q["is_impossible"]]
        unanswerable = [q for q in questions if q["is_impossible"]]
        for question in rng.sample(answerable, ANSWERABLE_PER_ARTICLE):
            answers = sorted({a["text"].strip() for a in question["answers"]})
            cases.append(
                {
                    "case_id": f"squad_{question['id']}",
                    "question": question["question"].strip(),
                    "should_answer": True,
                    "document": document,
                    "answers": answers,
                }
            )
        for question in rng.sample(unanswerable, UNANSWERABLE_PER_ARTICLE):
            cases.append(
                {
                    "case_id": f"squad_{question['id']}",
                    "question": question["question"].strip(),
                    "should_answer": False,
                    "document": document,
                }
            )
    return articles, cases


def main() -> None:
    if not RAW_FILE.exists():
        raise SystemExit(f"Download SQuAD 2.0 dev-v2.0.json to {RAW_FILE} first.")
    raw = RAW_FILE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise SystemExit("dev-v2.0.json does not match the expected SQuAD 2.0 file.")

    articles, cases = sample(json.loads(raw)["data"])
    corpus = HELDOUT_DIR / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    for article in articles:
        (corpus / f"{slug(article['title'])}.txt").write_text(
            article_text(article), encoding="utf-8", newline="\n"
        )
    payload = {
        "description": (
            "Held-out test set sampled from the SQuAD 2.0 development set by "
            "experiments/build_heldout_set.py (seed "
            f"{SEED}). Never used to choose any setting. Answerable cases list the "
            "crowd-written answers; unanswerable cases were written to look answerable."
        ),
        "source": "https://rajpurkar.github.io/SQuAD-explorer/ (dev-v2.0.json)",
        "license": "CC BY-SA 4.0, Rajpurkar, Jia and Liang (2018), Know What You Don't Know",
        "articles": [a["title"] for a in articles],
        "cases": cases,
    }
    (HELDOUT_DIR / "cases.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    answerable = sum(c["should_answer"] for c in cases)
    print(f"{len(articles)} articles, {len(cases)} cases ({answerable} answerable)")
    for article in articles:
        print(" ", article["title"])


if __name__ == "__main__":
    main()
