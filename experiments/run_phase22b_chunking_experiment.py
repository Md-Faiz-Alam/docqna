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
from docqna.ingestion.chunking import (
    BaseChunker,
    FixedChunker,
    SemanticChunker,
    StructureAwareChunker,
)
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, RRFHybridRetriever
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from docqna.settings import get_settings

StrategyName = Literal["fixed", "structure_aware", "semantic"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "sample.pdf"

DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_dataset_v2.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"

WORK_DIR = PROJECT_ROOT / "data" / "evaluation" / "work" / "phase22b"


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """One chunking configuration in the Phase 22B matrix."""

    name: StrategyName
    chunk_size: int = 200
    overlap: int = 40
    min_chunk_size: int | None = None
    similarity_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class ChunkStatistics:
    """Basic descriptive statistics for one chunk corpus."""

    chunk_count: int
    average_words: float
    min_words: int
    max_words: int


class TimedRetriever(BaseRetriever):
    """Transparent retrieval wrapper used for benchmark timing."""

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
        """Return average retrieval latency per query."""

        if self.call_count <= 0:
            return 0.0

        return self.total_seconds / self.call_count


def strategy_specs() -> list[StrategySpec]:
    """Return the fixed Phase 22B chunking comparison matrix."""

    return [
        StrategySpec(name="fixed", chunk_size=200, overlap=40),
        StrategySpec(name="structure_aware", chunk_size=200, overlap=40),
        StrategySpec(
            name="semantic",
            chunk_size=200,
            overlap=40,
            min_chunk_size=80,
            similarity_threshold=0.45,
        ),
    ]


def build_chunker(spec: StrategySpec, embedding_model: BaseEmbeddingModel) -> BaseChunker:
    """Build exactly one configured chunking strategy."""

    if spec.name == "fixed":
        return FixedChunker(chunk_size=spec.chunk_size, overlap=spec.overlap)

    if spec.name == "structure_aware":
        return StructureAwareChunker(chunk_size=spec.chunk_size, overlap=spec.overlap)

    if spec.name == "semantic":
        if spec.min_chunk_size is None or spec.similarity_threshold is None:
            raise ValueError("Semantic strategy requires min_chunk_size and similarity_threshold.")

        return SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=spec.chunk_size,
            overlap=spec.overlap,
            min_chunk_size=spec.min_chunk_size,
            similarity_threshold=(spec.similarity_threshold),
        )

    raise ValueError(f"Unsupported strategy: {spec.name!r}.")


def chunk_statistics(chunks: list[DocumentChunk]) -> ChunkStatistics:
    """Calculate simple word-count statistics."""

    if not chunks:
        raise ValueError("Chunk statistics require at least one chunk.")

    word_counts = [len(chunk.text.split()) for chunk in chunks]

    return ChunkStatistics(
        chunk_count=len(chunks),
        average_words=(sum(word_counts) / len(word_counts)),
        min_words=min(word_counts),
        max_words=max(word_counts),
    )


def build_retrieval_stack(
    *,
    chunks: list[DocumentChunk],
    embedding_model: BaseEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> BaseRetriever:
    """
    Rebuild the exact production retrieval architecture.

    Production configuration:
    dense + BM25 -> RRF(k=60, pool=10)
    -> CrossEncoder(pool=20, CPU)
    """

    dense_retriever = DenseRetriever(embedding_model=embedding_model, vector_store=vector_store)

    bm25_retriever = BM25Retriever(chunks=chunks)

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


def prepare_index(
    *, spec: StrategySpec, chunks: list[DocumentChunk], embedding_model: BaseEmbeddingModel
) -> tuple[ChromaVectorStore, float]:
    """Create a clean isolated Chroma index for one strategy."""

    index_dir = WORK_DIR / "indexes" / spec.name

    if index_dir.exists():
        shutil.rmtree(index_dir)

    index_dir.mkdir(parents=True, exist_ok=True)

    vector_store = ChromaVectorStore(
        persist_directory=index_dir, collection_name=(f"phase22b_{spec.name}")
    )

    started = time.perf_counter()

    embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

    vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

    elapsed = time.perf_counter() - started

    return (vector_store, elapsed)


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


def run_strategy(
    *,
    spec: StrategySpec,
    preprocessed_document,
    evaluation_cases,
    embedding_model: BaseEmbeddingModel,
    top_k: int,
) -> dict[str, object]:
    """Run one complete chunking + retrieval experiment."""

    print()
    print("=" * 72)
    print(f"Phase 22B strategy: {spec.name}")
    print("=" * 72)

    chunker = build_chunker(spec=spec, embedding_model=embedding_model)

    chunking_started = time.perf_counter()

    chunks = chunker.chunk(preprocessed_document)

    chunking_seconds = time.perf_counter() - chunking_started

    stats = chunk_statistics(chunks)

    print(
        "chunks="
        f"{stats.chunk_count} "
        "avg_words="
        f"{stats.average_words:.2f} "
        "min_words="
        f"{stats.min_words} "
        "max_words="
        f"{stats.max_words}"
    )

    vector_store, indexing_seconds = prepare_index(
        spec=spec, chunks=chunks, embedding_model=embedding_model
    )

    print(f"chunking_seconds={chunking_seconds:.3f} indexing_seconds={indexing_seconds:.3f}")

    retrieval_stack = build_retrieval_stack(
        chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    timed_retriever = TimedRetriever(retrieval_stack)

    evaluator = RetrievalEvaluator(timed_retriever)

    report = evaluator.evaluate(cases=evaluation_cases, top_k=top_k)

    metrics = report_metrics(report)

    metadata: dict[str, object] = {
        "phase": "22B",
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "strategy": spec.name,
        "chunk_size": spec.chunk_size,
        "overlap": spec.overlap,
        "min_chunk_size": (spec.min_chunk_size),
        "similarity_threshold": (spec.similarity_threshold),
        "chunk_count": stats.chunk_count,
        "average_chunk_words": (stats.average_words),
        "min_chunk_words": stats.min_words,
        "max_chunk_words": stats.max_words,
        "chunking_seconds": (chunking_seconds),
        "indexing_seconds": (indexing_seconds),
        "retrieval_total_seconds": (timed_retriever.total_seconds),
        "mean_query_retrieval_seconds": (timed_retriever.mean_seconds),
        "retrieval_architecture": ("dense+bm25->rrf(k=60,pool=10)->cross_encoder(pool=20,cpu)"),
    }

    save_report_bundle(
        report, RESULTS_DIR, stem=(f"phase22b_{spec.name}_retrieval"), metadata=metadata
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

    # Release model/index references before the next isolated strategy.
    del evaluator
    del timed_retriever
    del retrieval_stack
    del vector_store
    gc.collect()

    return result


def save_comparison(results: list[dict[str, object]], *, top_k: int) -> None:
    """Save combined Phase 22B JSON and CSV comparison tables."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "phase22b_chunking_comparison.json"

    csv_path = RESULTS_DIR / "phase22b_chunking_comparison.csv"

    payload = {
        "phase": "22B",
        "experiment": ("fixed_vs_structure_aware_vs_semantic"),
        "sample": str(SAMPLE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "top_k": top_k,
        "retrieval_architecture": {
            "dense_embedding": ("sentence-transformers/all-MiniLM-L6-v2"),
            "sparse": "BM25(k1=1.5,b=0.75)",
            "fusion": ("RRF(k=60,candidate_pool=10)"),
            "reranker": ("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "reranker_candidate_pool": 20,
            "reranker_device": "cpu",
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
        description=("Run the DocQ&A Phase 22B chunking strategy retrieval experiment.")
    )

    parser.add_argument(
        "--strategy",
        choices=["all", "fixed", "structure_aware", "semantic"],
        default="all",
        help=("Chunking strategy to evaluate. Default: all."),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=("Final retrieval depth. Must be at least 5. Default: 5."),
    )

    return parser.parse_args()


def main() -> None:
    """Run Phase 22B."""

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

    print("Loading and preprocessing sample.pdf once...")

    ingestion_service = DocumentIngestionService()

    document = ingestion_service.load(SAMPLE_PATH)

    preprocessed_document = ingestion_service.preprocess(document)

    evaluation_cases = load_evaluation_cases(DATASET_PATH)

    selected_specs = strategy_specs()

    if args.strategy != "all":
        selected_specs = [spec for spec in selected_specs if spec.name == args.strategy]

    results: list[dict[str, object]] = []

    for spec in selected_specs:
        result = run_strategy(
            spec=spec,
            preprocessed_document=(preprocessed_document),
            evaluation_cases=(evaluation_cases),
            embedding_model=(embedding_model),
            top_k=args.top_k,
        )

        results.append(result)

    # The combined comparison is most meaningful after all three
    # strategies have been run together under one environment.
    if args.strategy == "all":
        save_comparison(results, top_k=args.top_k)

    print()
    print("Phase 22B experiment completed.")


if __name__ == "__main__":
    main()
