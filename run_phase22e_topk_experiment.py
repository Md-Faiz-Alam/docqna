import csv
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    page_recall_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from docqna.evaluation.models import EvaluationCase
from docqna.evaluation.runner import load_evaluation_cases
from docqna.indexing import ChromaVectorStore
from docqna.ingestion import DocumentIngestionService
from docqna.ingestion.chunking import FixedChunker
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RRFHybridRetriever,
)
from docqna.retrieval.base import BaseRetriever
from docqna.settings import get_settings


PROJECT_ROOT = Path(
    __file__
).resolve().parent

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "sample.pdf"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_dataset_v2.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
)

WORK_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "work"
    / "phase22e"
)


@dataclass(frozen=True, slots=True)
class TopKCaseMetrics:
    """Metrics for one evaluation case at one final top-k."""

    case_id: str
    top_k: int
    hit_at_k: float
    reciprocal_rank_at_k: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    page_recall_at_k: float
    returned_results: int
    retrieval_seconds: float


def top_k_values() -> list[int]:
    """
    Final context sizes for Phase 22E.

    All values are <= the CrossEncoder candidate pool (20), so the
    upstream candidate set stays constant and only the final retained
    context depth changes.
    """

    return [
        3,
        5,
        10,
        20,
    ]


def prepare_fixed_chunks() -> list[DocumentChunk]:
    """Build the frozen production fixed 200/40 corpus."""

    service = DocumentIngestionService(
        chunker=FixedChunker(
            chunk_size=200,
            overlap=40,
        )
    )

    return service.process(
        SAMPLE_PATH
    ).chunks


def prepare_vector_store(
    *,
    chunks: list[DocumentChunk],
    embedding_model: SentenceTransformerEmbedding,
) -> tuple[
    ChromaVectorStore,
    float,
]:
    """Build one isolated Phase 22E vector index."""

    index_dir = (
        WORK_DIR
        / "index"
    )

    if index_dir.exists():
        shutil.rmtree(
            index_dir
        )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = ChromaVectorStore(
        persist_directory=index_dir,
        collection_name="phase22e_fixed",
    )

    started = time.perf_counter()

    embeddings = (
        embedding_model.embed_documents(
            [
                chunk.text
                for chunk in chunks
            ]
        )
    )

    vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    return (
        vector_store,
        elapsed,
    )


def build_production_retriever(
    *,
    chunks: list[DocumentChunk],
    embedding_model: SentenceTransformerEmbedding,
    vector_store: ChromaVectorStore,
) -> BaseRetriever:
    """Build the frozen production retrieval stack."""

    dense = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    bm25 = BM25Retriever(
        chunks=chunks,
        k1=1.5,
        b=0.75,
    )

    hybrid = RRFHybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
        rrf_k=60,
        candidate_pool_size=10,
    )

    return CrossEncoderReranker(
        base_retriever=hybrid,
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        candidate_pool_size=20,
        device="cpu",
    )


def evaluate_case(
    *,
    retriever: BaseRetriever,
    case: EvaluationCase,
    top_k: int,
) -> TopKCaseMetrics:
    """Evaluate one answerable case at one final context size."""

    started = time.perf_counter()

    results = retriever.retrieve(
        query=case.question,
        top_k=top_k,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    relevant_pages = set(
        case.relevant_pages
    )

    relevant_chunk_ids = set(
        case.relevant_chunk_ids
    )

    return TopKCaseMetrics(
        case_id=case.case_id,
        top_k=top_k,
        hit_at_k=hit_at_k(
            results,
            relevant_pages,
            top_k,
            relevant_chunk_ids,
        ),
        reciprocal_rank_at_k=(
            reciprocal_rank(
                results,
                relevant_pages,
                top_k,
                relevant_chunk_ids,
            )
        ),
        precision_at_k=precision_at_k(
            results,
            relevant_pages,
            top_k,
            relevant_chunk_ids,
        ),
        recall_at_k=recall_at_k(
            results,
            relevant_pages,
            top_k,
            relevant_chunk_ids,
        ),
        ndcg_at_k=ndcg_at_k(
            results,
            relevant_pages,
            top_k,
            relevant_chunk_ids,
        ),
        page_recall_at_k=page_recall_at_k(
            results,
            relevant_pages,
            top_k,
        ),
        returned_results=len(
            results
        ),
        retrieval_seconds=elapsed,
    )


def summarize(
    *,
    top_k: int,
    cases: list[TopKCaseMetrics],
) -> dict[str, float | int]:
    """Aggregate one top-k experiment slice."""

    if not cases:
        raise ValueError(
            "Cannot summarize an empty case list."
        )

    count = len(
        cases
    )

    return {
        "top_k": top_k,
        "evaluated_cases": count,
        "mean_hit_at_k": (
            sum(
                item.hit_at_k
                for item in cases
            )
            / count
        ),
        "mean_reciprocal_rank_at_k": (
            sum(
                item.reciprocal_rank_at_k
                for item in cases
            )
            / count
        ),
        "mean_precision_at_k": (
            sum(
                item.precision_at_k
                for item in cases
            )
            / count
        ),
        "mean_recall_at_k": (
            sum(
                item.recall_at_k
                for item in cases
            )
            / count
        ),
        "mean_ndcg_at_k": (
            sum(
                item.ndcg_at_k
                for item in cases
            )
            / count
        ),
        "mean_page_recall_at_k": (
            sum(
                item.page_recall_at_k
                for item in cases
            )
            / count
        ),
        "mean_returned_results": (
            sum(
                item.returned_results
                for item in cases
            )
            / count
        ),
        "mean_query_retrieval_seconds": (
            sum(
                item.retrieval_seconds
                for item in cases
            )
            / count
        ),
        "retrieval_total_seconds": sum(
            item.retrieval_seconds
            for item in cases
        ),
    }


def save_results(
    *,
    summaries: list[dict[str, float | int]],
    case_metrics: list[TopKCaseMetrics],
    indexing_seconds: float,
) -> None:
    """Write deterministic JSON and CSV outputs."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        RESULTS_DIR
        / "phase22e_topk_comparison.json"
    )

    summary_csv_path = (
        RESULTS_DIR
        / "phase22e_topk_comparison.csv"
    )

    cases_csv_path = (
        RESULTS_DIR
        / "phase22e_topk_cases.csv"
    )

    payload = {
        "phase": "22E",
        "experiment": "final_context_top_k",
        "sample": str(
            SAMPLE_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "dataset": str(
            DATASET_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "controlled_variables": {
            "chunking_strategy": "fixed",
            "chunk_size": 200,
            "overlap": 40,
            "embedding_model": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
            },
            "rrf": {
                "k": 60,
                "candidate_pool": 10,
            },
            "reranker": {
                "model": (
                    "cross-encoder/"
                    "ms-marco-MiniLM-L-6-v2"
                ),
                "candidate_pool": 20,
                "device": "cpu",
            },
            "indexing_seconds": (
                indexing_seconds
            ),
        },
        "top_k_values": top_k_values(),
        "summaries": summaries,
        "cases": [
            {
                "case_id": item.case_id,
                "top_k": item.top_k,
                "hit_at_k": item.hit_at_k,
                "reciprocal_rank_at_k": (
                    item.reciprocal_rank_at_k
                ),
                "precision_at_k": (
                    item.precision_at_k
                ),
                "recall_at_k": item.recall_at_k,
                "ndcg_at_k": item.ndcg_at_k,
                "page_recall_at_k": (
                    item.page_recall_at_k
                ),
                "returned_results": (
                    item.returned_results
                ),
                "retrieval_seconds": (
                    item.retrieval_seconds
                ),
            }
            for item in case_metrics
        ],
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if summaries:
        with summary_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    summaries[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                summaries
            )

    with cases_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "case_id",
            "top_k",
            "hit_at_k",
            "reciprocal_rank_at_k",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "page_recall_at_k",
            "returned_results",
            "retrieval_seconds",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in case_metrics:
            writer.writerow(
                {
                    "case_id": item.case_id,
                    "top_k": item.top_k,
                    "hit_at_k": item.hit_at_k,
                    "reciprocal_rank_at_k": (
                        item.reciprocal_rank_at_k
                    ),
                    "precision_at_k": (
                        item.precision_at_k
                    ),
                    "recall_at_k": (
                        item.recall_at_k
                    ),
                    "ndcg_at_k": item.ndcg_at_k,
                    "page_recall_at_k": (
                        item.page_recall_at_k
                    ),
                    "returned_results": (
                        item.returned_results
                    ),
                    "retrieval_seconds": (
                        item.retrieval_seconds
                    ),
                }
            )

    print()
    print(
        "Phase 22E outputs saved:"
    )
    print(
        f"  {json_path}"
    )
    print(
        f"  {summary_csv_path}"
    )
    print(
        f"  {cases_csv_path}"
    )


def main() -> None:
    """Run the controlled final-context top-k experiment."""

    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(
            f"Sample PDF not found: {SAMPLE_PATH}"
        )

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Phase 21 evaluation dataset not found: "
            f"{DATASET_PATH}"
        )

    settings = get_settings()

    print(
        "Preparing fixed 200/40 chunks..."
    )

    chunks = prepare_fixed_chunks()

    print(
        f"  chunks={len(chunks)}"
    )

    print(
        "Loading production embedding model..."
    )

    embedding_model = (
        SentenceTransformerEmbedding(
            model_name=(
                settings.embedding_model
            ),
            device=(
                settings.embedding_device
            ),
            batch_size=(
                settings.embedding_batch_size
            ),
        )
    )

    print(
        "Building isolated Phase 22E index..."
    )

    vector_store, indexing_seconds = (
        prepare_vector_store(
            chunks=chunks,
            embedding_model=embedding_model,
        )
    )

    print(
        "  indexing_seconds="
        f"{indexing_seconds:.3f}"
    )

    print(
        "Loading production CrossEncoder..."
    )

    retriever = build_production_retriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    all_cases = load_evaluation_cases(
        DATASET_PATH
    )

    answerable_cases = [
        case
        for case in all_cases
        if case.should_answer
    ]

    if not answerable_cases:
        raise ValueError(
            "Phase 22E requires answerable cases."
        )

    all_case_metrics: list[
        TopKCaseMetrics
    ] = []

    summaries: list[
        dict[str, float | int]
    ] = []

    for top_k in top_k_values():
        print()
        print(
            "=" * 72
        )
        print(
            f"Phase 22E final context top_k={top_k}"
        )
        print(
            "=" * 72
        )

        current_metrics: list[
            TopKCaseMetrics
        ] = []

        for case in answerable_cases:
            metrics = evaluate_case(
                retriever=retriever,
                case=case,
                top_k=top_k,
            )

            current_metrics.append(
                metrics
            )

        summary = summarize(
            top_k=top_k,
            cases=current_metrics,
        )

        summaries.append(
            summary
        )

        all_case_metrics.extend(
            current_metrics
        )

        print(
            "Hit@K="
            f"{summary['mean_hit_at_k']:.4f} "
            "MRR@K="
            f"{summary['mean_reciprocal_rank_at_k']:.4f} "
            "Precision@K="
            f"{summary['mean_precision_at_k']:.4f}"
        )

        print(
            "Recall@K="
            f"{summary['mean_recall_at_k']:.4f} "
            "nDCG@K="
            f"{summary['mean_ndcg_at_k']:.4f} "
            "PageRecall@K="
            f"{summary['mean_page_recall_at_k']:.4f}"
        )

        print(
            "mean_query_retrieval_seconds="
            f"{summary['mean_query_retrieval_seconds']:.4f}"
        )

    save_results(
        summaries=summaries,
        case_metrics=all_case_metrics,
        indexing_seconds=indexing_seconds,
    )

    del retriever
    del vector_store
    del embedding_model
    gc.collect()

    print()
    print(
        "Phase 22E experiment completed."
    )


if __name__ == "__main__":
    main()
