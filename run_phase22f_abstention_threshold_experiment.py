import csv
import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation.answerability_metrics import (
    f1_score,
    safe_divide,
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
    RetrievalConfidenceGate,
    RRFHybridRetriever,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
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
    / "phase22f"
)


@dataclass(frozen=True, slots=True)
class ScoredCase:
    """One evaluation case with its rank-1 cross-encoder score."""

    case_id: str
    question: str
    should_answer: bool
    top_score: float | None
    second_score: float | None
    score_margin: float | None
    result_count: int
    retrieval_seconds: float
    results: list[RetrievalResult]


def threshold_values() -> list[float]:
    """
    Thresholds for the Phase 22F calibration sweep.

    The current production threshold 0.0 is explicitly included.
    Values on both sides probe false-answer and false-abstention
    behavior against the labeled answerable/unanswerable dataset.
    """

    return [
        -10.0,
        -8.0,
        -5.0,
        0.0,
        1.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]


def prepare_fixed_chunks() -> list[DocumentChunk]:
    """Build the frozen production fixed 200/40 chunk corpus."""

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
    """Create one isolated Phase 22F Chroma index."""

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
        collection_name="phase22f_fixed",
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


def score_cases(
    *,
    retriever: BaseRetriever,
    cases: list[EvaluationCase],
    top_k: int,
) -> list[ScoredCase]:
    """Retrieve once per case and capture the gate's score signal."""

    scored_cases: list[
        ScoredCase
    ] = []

    score_reader = RetrievalConfidenceGate(
        min_top_score=0.0
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}] "
            f"{case.case_id} - retrieving confidence score..."
        )

        started = time.perf_counter()

        results = retriever.retrieve(
            query=case.question,
            top_k=top_k,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        decision = score_reader.evaluate(
            results
        )

        scored_cases.append(
            ScoredCase(
                case_id=case.case_id,
                question=case.question,
                should_answer=case.should_answer,
                top_score=decision.top_score,
                second_score=decision.second_score,
                score_margin=decision.score_margin,
                result_count=len(results),
                retrieval_seconds=elapsed,
                results=results,
            )
        )

    return scored_cases


def evaluate_threshold(
    *,
    threshold: float,
    scored_cases: list[ScoredCase],
) -> dict[str, float | int]:
    """Evaluate one confidence threshold using production gate logic."""

    gate = RetrievalConfidenceGate(
        min_top_score=threshold
    )

    true_answers = 0
    true_abstentions = 0
    false_answers = 0
    false_abstentions = 0

    for case in scored_cases:
        decision = gate.evaluate(
            case.results
        )

        predicted_should_answer = (
            decision.is_confident
        )

        if (
            case.should_answer
            and predicted_should_answer
        ):
            true_answers += 1

        elif (
            not case.should_answer
            and not predicted_should_answer
        ):
            true_abstentions += 1

        elif (
            not case.should_answer
            and predicted_should_answer
        ):
            false_answers += 1

        else:
            false_abstentions += 1

    total = len(
        scored_cases
    )

    accuracy = safe_divide(
        true_answers
        + true_abstentions,
        total,
    )

    answer_precision = safe_divide(
        true_answers,
        true_answers
        + false_answers,
    )

    answer_recall = safe_divide(
        true_answers,
        true_answers
        + false_abstentions,
    )

    answer_f1 = f1_score(
        answer_precision,
        answer_recall,
    )

    abstention_precision = safe_divide(
        true_abstentions,
        true_abstentions
        + false_abstentions,
    )

    abstention_recall = safe_divide(
        true_abstentions,
        true_abstentions
        + false_answers,
    )

    abstention_f1 = f1_score(
        abstention_precision,
        abstention_recall,
    )

    return {
        "threshold": threshold,
        "num_cases": total,
        "true_answers": true_answers,
        "true_abstentions": true_abstentions,
        "false_answers": false_answers,
        "false_abstentions": false_abstentions,
        "accuracy": accuracy,
        "answer_precision": answer_precision,
        "answer_recall": answer_recall,
        "answer_f1": answer_f1,
        "abstention_precision": (
            abstention_precision
        ),
        "abstention_recall": (
            abstention_recall
        ),
        "abstention_f1": (
            abstention_f1
        ),
    }


def score_summary(
    scored_cases: list[ScoredCase],
) -> dict[str, object]:
    """Summarize answerable and unanswerable rank-1 score ranges."""

    answerable_scores = [
        case.top_score
        for case in scored_cases
        if (
            case.should_answer
            and case.top_score is not None
        )
    ]

    unanswerable_scores = [
        case.top_score
        for case in scored_cases
        if (
            not case.should_answer
            and case.top_score is not None
        )
    ]

    def summary(
        values: list[float],
    ) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "max": None,
            }

        return {
            "count": len(values),
            "min": min(values),
            "mean": (
                sum(values)
                / len(values)
            ),
            "max": max(values),
        }

    answerable = summary(
        answerable_scores
    )

    unanswerable = summary(
        unanswerable_scores
    )

    score_gap: float | None = None

    answerable_min = answerable[
        "min"
    ]

    unanswerable_max = unanswerable[
        "max"
    ]

    if (
        isinstance(
            answerable_min,
            float,
        )
        and isinstance(
            unanswerable_max,
            float,
        )
    ):
        score_gap = (
            answerable_min
            - unanswerable_max
        )

    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
        "separation_gap": score_gap,
    }


def save_results(
    *,
    scored_cases: list[ScoredCase],
    threshold_results: list[
        dict[str, float | int]
    ],
    score_distribution: dict[str, object],
    indexing_seconds: float,
    top_k: int,
) -> None:
    """Save Phase 22F JSON and CSV artifacts."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        RESULTS_DIR
        / "phase22f_abstention_threshold_comparison.json"
    )

    threshold_csv_path = (
        RESULTS_DIR
        / "phase22f_abstention_threshold_comparison.csv"
    )

    scores_csv_path = (
        RESULTS_DIR
        / "phase22f_case_scores.csv"
    )

    payload = {
        "phase": "22F",
        "experiment": (
            "abstention_threshold_calibration"
        ),
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
        "top_k": top_k,
        "production_threshold": 0.0,
        "controlled_variables": {
            "chunking_strategy": "fixed",
            "chunk_size": 200,
            "overlap": 40,
            "embedding_model": (
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            ),
            "retrieval_architecture": (
                "dense+bm25->rrf(k=60,pool=10)"
                "->cross_encoder(pool=20,cpu)"
            ),
            "confidence_signal": (
                "rank_1_cross_encoder_score"
            ),
            "indexing_seconds": (
                indexing_seconds
            ),
        },
        "score_distribution": (
            score_distribution
        ),
        "threshold_results": (
            threshold_results
        ),
        "cases": [
            {
                "case_id": case.case_id,
                "question": case.question,
                "should_answer": (
                    case.should_answer
                ),
                "top_score": case.top_score,
                "second_score": (
                    case.second_score
                ),
                "score_margin": (
                    case.score_margin
                ),
                "result_count": (
                    case.result_count
                ),
                "retrieval_seconds": (
                    case.retrieval_seconds
                ),
            }
            for case in scored_cases
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

    with threshold_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                threshold_results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            threshold_results
        )

    with scores_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = [
            "case_id",
            "question",
            "should_answer",
            "top_score",
            "second_score",
            "score_margin",
            "result_count",
            "retrieval_seconds",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for case in scored_cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "should_answer": (
                        case.should_answer
                    ),
                    "top_score": case.top_score,
                    "second_score": (
                        case.second_score
                    ),
                    "score_margin": (
                        case.score_margin
                    ),
                    "result_count": (
                        case.result_count
                    ),
                    "retrieval_seconds": (
                        case.retrieval_seconds
                    ),
                }
            )

    print()
    print(
        "Phase 22F outputs saved:"
    )
    print(
        f"  {json_path}"
    )
    print(
        f"  {threshold_csv_path}"
    )
    print(
        f"  {scores_csv_path}"
    )


def main() -> None:
    """Run Phase 22F confidence-threshold calibration."""

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
        "Preparing frozen fixed 200/40 chunks..."
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
        "Building isolated Phase 22F index..."
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
        "Loading production retrieval stack..."
    )

    retriever = build_production_retriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    cases = load_evaluation_cases(
        DATASET_PATH
    )

    top_k = 5

    print()
    print(
        "Retrieving rank-1 confidence scores..."
    )

    scored_cases = score_cases(
        retriever=retriever,
        cases=cases,
        top_k=top_k,
    )

    distribution = score_summary(
        scored_cases
    )

    print()
    print(
        "Score distribution:"
    )
    print(
        "  answerable="
        f"{distribution['answerable']}"
    )
    print(
        "  unanswerable="
        f"{distribution['unanswerable']}"
    )
    print(
        "  separation_gap="
        f"{distribution['separation_gap']}"
    )

    threshold_results: list[
        dict[str, float | int]
    ] = []

    print()
    print(
        "Threshold sweep:"
    )

    for threshold in threshold_values():
        result = evaluate_threshold(
            threshold=threshold,
            scored_cases=scored_cases,
        )

        threshold_results.append(
            result
        )

        print(
            "  threshold="
            f"{threshold:>5.1f} "
            "accuracy="
            f"{result['accuracy']:.4f} "
            "answer_f1="
            f"{result['answer_f1']:.4f} "
            "abstention_f1="
            f"{result['abstention_f1']:.4f} "
            "FP_answers="
            f"{result['false_answers']} "
            "FN_abstentions="
            f"{result['false_abstentions']}"
        )

    save_results(
        scored_cases=scored_cases,
        threshold_results=threshold_results,
        score_distribution=distribution,
        indexing_seconds=indexing_seconds,
        top_k=top_k,
    )

    del retriever
    del vector_store
    del embedding_model
    gc.collect()

    print()
    print(
        "Phase 22F experiment completed."
    )


if __name__ == "__main__":
    main()
