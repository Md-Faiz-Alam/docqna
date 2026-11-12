import argparse
import csv
import gc
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.embeddings.base import BaseEmbeddingModel
from docqna.evaluation import (
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    load_evaluation_cases,
    save_report_bundle,
)
from docqna.indexing import ChromaVectorStore
from docqna.ingestion import DocumentIngestionService
from docqna.ingestion.chunking import FixedChunker
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, RRFHybridRetriever
from docqna.retrieval.base import BaseRetriever
from docqna.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "sample.pdf"

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_dataset_v2.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

WORK_DIR = PROJECT_ROOT / "data" / "evaluation" / "work" / "phase22d"


@dataclass(frozen=True, slots=True)
class EmbeddingModelSpec:
    """One embedding-model configuration in Phase 22D."""

    label: str
    model_name: str
    role: str


@dataclass(frozen=True, slots=True)
class TimedRetrievalResult:
    """Aggregate timing for one evaluated retriever."""

    total_seconds: float
    mean_seconds: float


class TimedRetriever(BaseRetriever):
    """Transparent wrapper that records retrieval latency."""

    def __init__(self, delegate: BaseRetriever) -> None:
        self.delegate = delegate
        self.total_seconds = 0.0
        self.call_count = 0

    def retrieve(self, query: str, top_k: int = 5, metadata_filters=None):
        started = time.perf_counter()

        try:
            if metadata_filters:
                return self.delegate.retrieve(
                    query=query, top_k=top_k, metadata_filters=metadata_filters
                )

            return self.delegate.retrieve(query=query, top_k=top_k)

        finally:
            self.total_seconds += time.perf_counter() - started
            self.call_count += 1

    def timing(self) -> TimedRetrievalResult:
        """Return accumulated retrieval timing."""

        mean_seconds = self.total_seconds / self.call_count if self.call_count else 0.0

        return TimedRetrievalResult(total_seconds=self.total_seconds, mean_seconds=mean_seconds)


def embedding_model_specs() -> list[EmbeddingModelSpec]:
    """
    Return the controlled Phase 22D embedding-model matrix.

    These models all work directly with the project's existing
    SentenceTransformerEmbedding wrapper without query/document
    prefix changes.
    """

    return [
        EmbeddingModelSpec(
            label="all_minilm_l6_v2",
            model_name=("sentence-transformers/all-MiniLM-L6-v2"),
            role="production_baseline",
        ),
        EmbeddingModelSpec(
            label="multi_qa_minilm_l6_cos_v1",
            model_name=("sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
            role="retrieval_oriented_minilm",
        ),
        EmbeddingModelSpec(
            label="all_mpnet_base_v2",
            model_name=("sentence-transformers/all-mpnet-base-v2"),
            role="larger_quality_candidate",
        ),
    ]


def safe_collection_name(label: str) -> str:
    """Create a deterministic Chroma-safe collection name."""

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip()).strip("_")

    if not cleaned:
        raise ValueError("Embedding-model label cannot be empty.")

    return f"phase22d_{cleaned}"


def prepare_fixed_chunks() -> tuple[list[DocumentChunk], float]:
    """Build the frozen production fixed 200/40 chunk corpus."""

    ingestion_service = DocumentIngestionService(chunker=FixedChunker(chunk_size=200, overlap=40))

    started = time.perf_counter()

    result = ingestion_service.process(SAMPLE_PATH)

    elapsed = time.perf_counter() - started

    return (result.chunks, elapsed)


def build_production_retriever(
    *,
    chunks: list[DocumentChunk],
    embedding_model: BaseEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> BaseRetriever:
    """
    Build the frozen production retrieval architecture.

    Only the embedding model varies in Phase 22D.
    """

    dense_retriever = DenseRetriever(embedding_model=embedding_model, vector_store=vector_store)

    bm25_retriever = BM25Retriever(chunks=chunks, k1=1.5, b=0.75)

    hybrid_retriever = RRFHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=bm25_retriever,
        rrf_k=60,
        candidate_pool_size=10,
    )

    return CrossEncoderReranker(
        base_retriever=hybrid_retriever,
        model_name=("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        candidate_pool_size=20,
        device="cpu",
    )


def report_metrics(report: RetrievalEvaluationReport) -> dict[str, float | int]:
    """Convert retrieval aggregates into comparison fields."""

    return {
        "evaluated_cases": report.num_cases,
        "skipped_unanswerable": (report.num_skipped_unanswerable),
        "hit_at_1": report.mean_hit_at_1,
        "hit_at_3": report.mean_hit_at_3,
        "hit_at_5": report.mean_hit_at_5,
        "mrr_at_5": (report.mean_reciprocal_rank),
        "page_recall_at_5": (report.mean_page_recall_at_5),
        "precision_at_k": (report.mean_precision_at_k),
        "recall_at_k": (report.mean_recall_at_k),
        "ndcg_at_k": (report.mean_ndcg_at_k),
    }


def run_model(
    *,
    spec: EmbeddingModelSpec,
    chunks: list[DocumentChunk],
    evaluation_cases,
    device: str | None,
    batch_size: int,
    top_k: int,
) -> dict[str, object]:
    """Run one embedding model through the full production retriever."""

    print()
    print("=" * 72)
    print(f"Phase 22D embedding model: {spec.label}")
    print(f"model={spec.model_name}")
    print("=" * 72)

    model_started = time.perf_counter()

    embedding_model = SentenceTransformerEmbedding(
        model_name=spec.model_name, device=device, batch_size=batch_size
    )

    model_load_seconds = time.perf_counter() - model_started

    dimension = embedding_model.dimension

    index_dir = WORK_DIR / "indexes" / spec.label

    if index_dir.exists():
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    vector_store = ChromaVectorStore(
        persist_directory=index_dir, collection_name=(safe_collection_name(spec.label))
    )

    indexing_started = time.perf_counter()

    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    indexing_seconds = time.perf_counter() - indexing_started

    retriever = build_production_retriever(
        chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    timed_retriever = TimedRetriever(retriever)

    evaluator = RetrievalEvaluator(timed_retriever)

    report = evaluator.evaluate(cases=evaluation_cases, top_k=top_k)

    timing = timed_retriever.timing()

    metadata: dict[str, object] = {
        "phase": "22D",
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "embedding_label": spec.label,
        "embedding_model": spec.model_name,
        "embedding_role": spec.role,
        "embedding_dimension": dimension,
        "device": device,
        "batch_size": batch_size,
        "chunking_strategy": "fixed",
        "chunk_size": 200,
        "overlap": 40,
        "chunk_count": len(chunks),
        "model_load_seconds": (model_load_seconds),
        "indexing_seconds": (indexing_seconds),
        "retrieval_total_seconds": (timing.total_seconds),
        "mean_query_retrieval_seconds": (timing.mean_seconds),
        "retrieval_architecture": ("dense+bm25->rrf(k=60,pool=10)->cross_encoder(pool=20,cpu)"),
    }

    save_report_bundle(
        report, RESULTS_DIR, stem=(f"phase22d_{spec.label}_retrieval"), metadata=metadata
    )

    result: dict[str, object] = {**metadata, **report_metrics(report)}

    print(
        "dimension="
        f"{dimension} "
        "model_load_seconds="
        f"{model_load_seconds:.3f} "
        "indexing_seconds="
        f"{indexing_seconds:.3f}"
    )

    print(
        "Hit@1="
        f"{report.mean_hit_at_1:.4f} "
        "Hit@3="
        f"{report.mean_hit_at_3:.4f} "
        "Hit@5="
        f"{report.mean_hit_at_5:.4f} "
        "MRR@5="
        f"{report.mean_reciprocal_rank:.4f}"
    )

    print(
        "Precision@K="
        f"{report.mean_precision_at_k:.4f} "
        "Recall@K="
        f"{report.mean_recall_at_k:.4f} "
        "nDCG@K="
        f"{report.mean_ndcg_at_k:.4f} "
        "PageRecall@5="
        f"{report.mean_page_recall_at_5:.4f}"
    )

    print(f"mean_query_retrieval_seconds={timing.mean_seconds:.4f}")

    del evaluator
    del timed_retriever
    del retriever
    del vector_store
    del embeddings
    del embedding_model
    gc.collect()

    return result


def save_comparison(
    results: list[dict[str, object]], *, top_k: int, fixed_chunking_seconds: float
) -> None:
    """Save combined Phase 22D JSON and CSV comparison."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "phase22d_embedding_comparison.json"

    csv_path = RESULTS_DIR / "phase22d_embedding_comparison.csv"

    payload = {
        "phase": "22D",
        "experiment": ("embedding_model_comparison"),
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "top_k": top_k,
        "controlled_variables": {
            "chunking_strategy": "fixed",
            "chunk_size": 200,
            "overlap": 40,
            "retrieval_architecture": ("dense+bm25->rrf(k=60,pool=10)->cross_encoder(pool=20,cpu)"),
            "fixed_chunking_seconds": (fixed_chunking_seconds),
        },
        "results": results,
    }

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames: list[str] = []

    for result in results:
        for key in result:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print()
    print("Combined comparison saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def parse_args() -> argparse.Namespace:
    """Parse Phase 22D CLI arguments."""

    labels = [spec.label for spec in embedding_model_specs()]

    parser = argparse.ArgumentParser(
        description=("Run the DocQ&A Phase 22D embedding-model experiment.")
    )

    parser.add_argument(
        "--model",
        choices=["all", *labels],
        default="all",
        help=("Embedding model to evaluate. Default: all."),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=("Final retrieval depth. Must be at least 5. Default: 5."),
    )

    return parser.parse_args()


def main() -> None:
    """Run Phase 22D."""

    args = parse_args()

    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5.")

    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Sample PDF not found: {SAMPLE_PATH}")

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Phase 21 evaluation dataset not found: {DATASET_PATH}")

    settings = get_settings()

    print("Preparing frozen fixed 200/40 chunks...")

    chunks, fixed_chunking_seconds = prepare_fixed_chunks()

    print(f"  chunks={len(chunks)} chunking_seconds={fixed_chunking_seconds:.3f}")

    print("Experiment runtime:")
    print(f"  device={settings.embedding_device}")
    print(f"  batch_size={settings.embedding_batch_size}")

    evaluation_cases = load_evaluation_cases(DATASET_PATH)

    selected_specs = embedding_model_specs()

    if args.model != "all":
        selected_specs = [spec for spec in selected_specs if spec.label == args.model]

    results: list[dict[str, object]] = []

    for spec in selected_specs:
        result = run_model(
            spec=spec,
            chunks=chunks,
            evaluation_cases=(evaluation_cases),
            device=settings.embedding_device,
            batch_size=(settings.embedding_batch_size),
            top_k=args.top_k,
        )

        results.append(result)

    if args.model == "all":
        save_comparison(results, top_k=args.top_k, fixed_chunking_seconds=(fixed_chunking_seconds))

    print()
    print("Phase 22D experiment completed.")


if __name__ == "__main__":
    main()
