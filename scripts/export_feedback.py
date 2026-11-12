"""Export down-voted answers as draft evaluation cases.

Each exported case needs a human to fill in the expected evidence (or mark it
unanswerable) before it is added to an evaluation dataset.

    python scripts/export_feedback.py --out data/evaluation/feedback_drafts.json
"""

import argparse
import json
from pathlib import Path

from docqna.feedback import FeedbackStore
from docqna.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_settings().metadata_dir / "feedback.db")
    parser.add_argument("--rating", choices=["up", "down"], default="down")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = FeedbackStore(args.db, retention_days=0).list(rating=args.rating)
    drafts = [
        {
            "case_id": f"feedback_{record.id}",
            "question": record.question,
            "observed_answer": record.answer,
            "observed_abstained": record.abstained,
            "user_comment": record.comment,
            "should_answer": None,  # TODO(reviewer): true / false
            "expected_answer_points": [],  # TODO(reviewer)
            "expected_pages": [],  # TODO(reviewer)
        }
        for record in records
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(drafts)} draft cases to {args.out}")


if __name__ == "__main__":
    main()
