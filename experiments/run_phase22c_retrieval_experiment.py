import argparse
import csv
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.embeddings.base import BaseEmbeddingModel
from docqna.evaluation import (
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    load_evaluation_cases,
    save_report_bundle,
)
from docqna.filters import MetadataFilters
from docqna.indexing import ChromaVectorStore
from docqna.ingestion import DocumentIngestionService
from docqna.ingestion.chunking import FixedChunker
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, RRFHybridRetriever
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from docqna.settings import get_settings

RetrieverName = Literal["dense", "bm25", "rrf", "rrf_reranker"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "sample.pdf"

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_dataset_v2.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

WORK_DIR = PROJECT_ROOT / "data" / "evaluation" / "work" / "phase22c"


@dataclass(frozen=True, slots=True)
class RetrieverSpec:
    """One retrieval configuration in the Phase 22C matrix."""

    name: RetrieverName
    description: str


class TimedRetriever(BaseRetriever):
    """Transparent retrieval wrapper for per-query timing."""

    def __init__(self, delegate: BaseRetriever) -> None:
        self.delegate = delegate
        self.total_seconds = 0.0
        self.call_count = 0

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
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

    @property
    def mean_seconds(self) -> float:
        """Return mean retrieval latency per query."""

        if self.call_count <= 0:
            return 0.0

        return self.total_seconds / self.call_count


def retriever_specs() -> list[RetrieverSpec]:
    """Return the controlled Phase 22C retrieval matrix."""

    return [
        RetrieverSpec(
            name="dense", description=("Dense retrieval with all-MiniLM-L6-v2 embeddings")
        ),
        RetrieverSpec(name="bm25", description=("BM25 sparse retrieval (k1=1.5, b=0.75)")),
        RetrieverSpec(name="rrf", description=("Dense + BM25 with RRF (k=60, candidate_pool=10)")),
        RetrieverSpec(
            name="rrf_reranker",
            description=(
                "Dense + BM25 + RRF + CrossEncoder (RRF k=60, RRF pool=10, reranker pool=20)"
            ),
        ),
    ]


def prepare_fixed_chunks() -> tuple[list[DocumentChunk], float]:
    """Load, preprocess, and chunk sample.pdf with fixed 200/40."""

    ingestion_service = DocumentIngestionService(chunker=FixedChunker(chunk_size=200, overlap=40))

    started = time.perf_counter()

    result = ingestion_service.process(SAMPLE_PATH)

    elapsed = time.perf_counter() - started

    return (result.chunks, elapsed)


def prepare_index(
    *, chunks: list[DocumentChunk], embedding_model: BaseEmbeddingModel
) -> tuple[ChromaVectorStore, float]:
    """Create one isolated fixed-chunk Chroma index for Phase 22C."""

    index_dir = WORK_DIR / "index"

    if index_dir.exists():
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    vector_store = ChromaVectorStore(
        persist_directory=index_dir, collection_name=("phase22c_fixed")
    )

    started = time.perf_counter()

    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    elapsed = time.perf_counter() - started

    return (vector_store, elapsed)


def build_retriever(
    *,
    spec: RetrieverSpec,
    chunks: list[DocumentChunk],
    embedding_model: BaseEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> BaseRetriever:
    """Build exactly one retrieval architecture."""

    dense = DenseRetriever(embedding_model=embedding_model, vector_store=vector_store)

    bm25 = BM25Retriever(chunks=chunks, k1=1.5, b=0.75)

    if spec.name == "dense":
        return dense

    if spec.name == "bm25":
        return bm25

    hybrid = RRFHybridRetriever(
        dense_retriever=dense, sparse_retriever=bm25, rrf_k=60, candidate_pool_size=10
    )

    if spec.name == "rrf":
        return hybrid

    if spec.name == "rrf_reranker":
        return CrossEncoderReranker(
            base_retriever=hybrid,
            model_name=("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            candidate_pool_size=20,
            device="cpu",
        )

    raise ValueError(f"Unsupported retriever: {spec.name!r}.")


def report_metrics(report: RetrievalEvaluationReport) -> dict[str, float | int]:
    """Convert retrieval report aggregates into comparison columns."""

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


def run_retriever(
    *,
    spec: RetrieverSpec,
    chunks: list[DocumentChunk],
    embedding_model: BaseEmbeddingModel,
    vector_store: ChromaVectorStore,
    evaluation_cases,
    top_k: int,
) -> dict[str, object]:
    """Evaluate one retrieval architecture."""

    print()
    print("=" * 72)
    print(f"Phase 22C retriever: {spec.name}")
    print("=" * 72)

    build_started = time.perf_counter()

    retriever = build_retriever(
        spec=spec, chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    build_seconds = time.perf_counter() - build_started

    timed_retriever = TimedRetriever(retriever)

    evaluator = RetrievalEvaluator(timed_retriever)

    report = evaluator.evaluate(cases=evaluation_cases, top_k=top_k)

    metrics = report_metrics(report)

    metadata: dict[str, object] = {
        "phase": "22C",
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "retriever": spec.name,
        "description": spec.description,
        "chunking_strategy": "fixed",
        "chunk_size": 200,
        "overlap": 40,
        "chunk_count": len(chunks),
        "retriever_build_seconds": (build_seconds),
        "retrieval_total_seconds": (timed_retriever.total_seconds),
        "mean_query_retrieval_seconds": (timed_retriever.mean_seconds),
    }

    save_report_bundle(
        report, RESULTS_DIR, stem=(f"phase22c_{spec.name}_retrieval"), metadata=metadata
    )

    result: dict[str, object] = {**metadata, **metrics}

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

    print(f"mean_query_retrieval_seconds={timed_retriever.mean_seconds:.4f}")

    del evaluator
    del timed_retriever
    del retriever
    gc.collect()

    return result


def save_comparison(
    results: list[dict[str, object]],
    *,
    top_k: int,
    fixed_chunking_seconds: float,
    indexing_seconds: float,
) -> None:
    """Save combined Phase 22C JSON and CSV comparison tables."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "phase22c_retrieval_comparison.json"

    csv_path = RESULTS_DIR / "phase22c_retrieval_comparison.csv"

    payload = {
        "phase": "22C",
        "experiment": ("dense_vs_bm25_vs_rrf_vs_reranker"),
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "top_k": top_k,
        "fixed_chunking": {
            "chunk_size": 200,
            "overlap": 40,
            "chunking_seconds": (fixed_chunking_seconds),
            "indexing_seconds": (indexing_seconds),
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
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Run the DocQ&A Phase 22C retrieval architecture experiment.")
    )

    parser.add_argument(
        "--retriever",
        choices=["all", "dense", "bm25", "rrf", "rrf_reranker"],
        default="all",
        help=("Retrieval architecture to evaluate. Default: all."),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=("Final retrieval depth. Must be at least 5. Default: 5."),
    )

    return parser.parse_args()


def main() -> None:
    """Run Phase 22C."""

    args = parse_args()

    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5.")

    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(f"Sample PDF not found: {SAMPLE_PATH}")

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Phase 21 evaluation dataset not found: {DATASET_PATH}")

    settings = get_settings()

    print("Loading embedding model:")
    print(f"  model={settings.embedding_model}")
    print(f"  device={settings.embedding_device}")
    print(f"  batch_size={settings.embedding_batch_size}")

    embedding_model = SentenceTransformerEmbedding(
        model_name=(settings.embedding_model),
        device=(settings.embedding_device),
        batch_size=(settings.embedding_batch_size),
    )

    print("Preparing fixed 200/40 chunks...")

    chunks, fixed_chunking_seconds = prepare_fixed_chunks()

    print(f"  chunks={len(chunks)} chunking_seconds={fixed_chunking_seconds:.3f}")

    print("Building isolated Phase 22C Chroma index...")

    vector_store, indexing_seconds = prepare_index(chunks=chunks, embedding_model=embedding_model)

    print(f"  indexing_seconds={indexing_seconds:.3f}")

    evaluation_cases = load_evaluation_cases(DATASET_PATH)

    selected_specs = retriever_specs()

    if args.retriever != "all":
        selected_specs = [spec for spec in selected_specs if spec.name == args.retriever]

    results: list[dict[str, object]] = []

    for spec in selected_specs:
        result = run_retriever(
            spec=spec,
            chunks=chunks,
            embedding_model=(embedding_model),
            vector_store=vector_store,
            evaluation_cases=(evaluation_cases),
            top_k=args.top_k,
        )

        results.append(result)

    if args.retriever == "all":
        save_comparison(
            results,
            top_k=args.top_k,
            fixed_chunking_seconds=(fixed_chunking_seconds),
            indexing_seconds=(indexing_seconds),
        )

    print()
    print("Phase 22C experiment completed.")


if __name__ == "__main__":
    main()
