"""Public retrieval + answerability benchmark (runs in CI).

Indexes the fictional Northwind corpus in `data/evaluation/public/corpus/`
with the *production* pipeline (fixed chunking, MiniLM embeddings, dense +
per-session BM25, RRF, cross-encoder, confidence gate) and scores:

- retrieval on answerable cases: Hit@1, Hit@3, Hit@5, MRR
- answerability: how often answerable questions pass the confidence gate,
  and how often unanswerable questions are correctly abstained.

No LLM is involved, so it is deterministic and runs on CPU in CI.

    python -m docqna.evaluation.public_benchmark            # print report
    python -m docqna.evaluation.public_benchmark --check    # fail below thresholds
"""

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from docqna.constants import PROJECT_ROOT
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.confidence import RetrievalConfidenceGate
from docqna.retrieval.models import RetrievalResult

BENCHMARK_DIR = PROJECT_ROOT / "data" / "evaluation" / "public"
SESSION_ID = "public-benchmark-" + "0" * 32
TOP_K = 5


class BenchmarkCase(BaseModel):
    case_id: str
    question: str
    should_answer: bool = True
    document: str | None = None
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answerable_cases_need_evidence(self) -> "BenchmarkCase":
        if self.should_answer and (not self.document or not self.evidence):
            raise ValueError(f"{self.case_id}: answerable cases need document and evidence")
        if not self.should_answer and (self.document or self.evidence):
            raise ValueError(f"{self.case_id}: unanswerable cases must not name evidence")
        return self


class Thresholds(BaseModel):
    min_hit_at_1: float
    min_hit_at_5: float
    min_mrr: float
    min_answerable_pass_rate: float
    min_unanswerable_abstain_rate: float


class CaseResult(BaseModel):
    case_id: str
    should_answer: bool
    first_relevant_rank: int | None
    top_score: float | None
    gate_passed: bool


class BenchmarkReport(BaseModel):
    num_answerable: int
    num_unanswerable: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    answerable_pass_rate: float
    unanswerable_abstain_rate: float
    cases: list[CaseResult]

    def failures(self, thresholds: Thresholds) -> list[str]:
        checks = [
            ("Hit@1", self.hit_at_1, thresholds.min_hit_at_1),
            ("Hit@5", self.hit_at_5, thresholds.min_hit_at_5),
            ("MRR", self.mrr, thresholds.min_mrr),
            (
                "answerable pass rate",
                self.answerable_pass_rate,
                thresholds.min_answerable_pass_rate,
            ),
            (
                "unanswerable abstain rate",
                self.unanswerable_abstain_rate,
                thresholds.min_unanswerable_abstain_rate,
            ),
        ]
        return [
            f"{name} {value:.3f} < required {minimum:.3f}"
            for name, value, minimum in checks
            if value < minimum
        ]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_cases(path: Path = BENCHMARK_DIR / "cases.json") -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [BenchmarkCase.model_validate(item) for item in raw["cases"]]
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case_id in benchmark cases.")
    return cases


def load_thresholds(path: Path = BENCHMARK_DIR / "thresholds.json") -> Thresholds:
    return Thresholds.model_validate_json(path.read_text(encoding="utf-8"))


def is_relevant(result: RetrievalResult, case: BenchmarkCase) -> bool:
    if result.filename != case.document:
        return False
    text = _normalise(result.text)
    return any(_normalise(phrase) in text for phrase in case.evidence)


def score_cases(
    cases: list[BenchmarkCase], retriever: BaseRetriever, gate: RetrievalConfidenceGate
) -> BenchmarkReport:
    results: list[CaseResult] = []
    for case in cases:
        retrieved = retriever.retrieve(
            case.question, top_k=TOP_K, metadata_filters={"session_id": SESSION_ID}
        )
        first_relevant = next(
            (r.rank for r in retrieved if case.should_answer and is_relevant(r, case)), None
        )
        decision = gate.evaluate(retrieved)
        results.append(
            CaseResult(
                case_id=case.case_id,
                should_answer=case.should_answer,
                first_relevant_rank=first_relevant,
                top_score=decision.top_score,
                gate_passed=decision.is_confident,
            )
        )

    answerable = [r for r in results if r.should_answer]
    unanswerable = [r for r in results if not r.should_answer]

    def hit(k: int) -> float:
        hits = [r for r in answerable if r.first_relevant_rank and r.first_relevant_rank <= k]
        return len(hits) / len(answerable) if answerable else 0.0

    return BenchmarkReport(
        num_answerable=len(answerable),
        num_unanswerable=len(unanswerable),
        hit_at_1=hit(1),
        hit_at_3=hit(3),
        hit_at_5=hit(5),
        mrr=(
            sum(1.0 / r.first_relevant_rank for r in answerable if r.first_relevant_rank)
            / len(answerable)
            if answerable
            else 0.0
        ),
        answerable_pass_rate=(
            sum(r.gate_passed for r in answerable) / len(answerable) if answerable else 0.0
        ),
        unanswerable_abstain_rate=(
            sum(not r.gate_passed for r in unanswerable) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
        cases=results,
    )


def run_benchmark(work_dir: Path) -> BenchmarkReport:
    """Index the public corpus with production components and score all cases."""

    from docqna.api.dependencies import build_session_retriever
    from docqna.embeddings import SentenceTransformerEmbedding
    from docqna.indexing import ChromaVectorStore, DocumentIndexService
    from docqna.ingestion import DocumentIngestionService, DocumentRegistry
    from docqna.retrieval import CrossEncoderReranker
    from docqna.settings import Settings

    settings = Settings(_env_file=None)  # type: ignore[call-arg]  # ignore local .env
    embedding_model = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(
        persist_directory=work_dir / "chroma", collection_name="public_benchmark"
    )
    index_service = DocumentIndexService(
        ingestion_service=DocumentIngestionService(),
        registry=DocumentRegistry(work_dir / "registry.db"),
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    for document in sorted((BENCHMARK_DIR / "corpus").glob("*.txt")):
        index_service.index_document(document, session_id=SESSION_ID)

    retriever = CrossEncoderReranker(
        base_retriever=build_session_retriever(
            settings=settings, embedding_model=embedding_model, vector_store=vector_store
        ),
        model_name=settings.reranker_model,
        candidate_pool_size=settings.reranker_candidate_pool_size,
        device=settings.reranker_device,
    )
    gate = RetrievalConfidenceGate(min_top_score=settings.answerability_min_score)
    return score_cases(load_cases(), retriever, gate)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public DocQ&A benchmark.")
    parser.add_argument("--check", action="store_true", help="exit 1 if below thresholds")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "results" / "public_benchmark.json",
    )
    args = parser.parse_args()

    work_dir = Path(tempfile.mkdtemp(prefix="docqna-benchmark-"))
    try:
        report = run_benchmark(work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(
        f"answerable={report.num_answerable} unanswerable={report.num_unanswerable}\n"
        f"Hit@1={report.hit_at_1:.3f} Hit@3={report.hit_at_3:.3f} "
        f"Hit@5={report.hit_at_5:.3f} MRR={report.mrr:.3f}\n"
        f"answerable pass rate={report.answerable_pass_rate:.3f} "
        f"unanswerable abstain rate={report.unanswerable_abstain_rate:.3f}"
    )
    misses = [c for c in report.cases if c.should_answer and c.first_relevant_rank != 1]
    wrong_gate = [c for c in report.cases if c.gate_passed != c.should_answer]
    for case in misses:
        print(
            f"  retrieval miss@1: {case.case_id} (first relevant rank {case.first_relevant_rank})"
        )
    for case in wrong_gate:
        print(f"  gate wrong: {case.case_id} top_score={case.top_score}")

    if args.check:
        failures = report.failures(load_thresholds())
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
