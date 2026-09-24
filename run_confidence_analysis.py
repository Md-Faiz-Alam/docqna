import json
import statistics
from pathlib import Path
from typing import Any

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation import load_evaluation_cases
from docqna.indexing import ChromaVectorStore
from docqna.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RRFHybridRetriever,
)
from docqna.settings import get_settings


settings = get_settings()


# --------------------------------
# 1. Paths
# --------------------------------

answerable_file = Path(
    "data/evaluation/baseline_questions.json"
)

unanswerable_file = Path(
    "data/evaluation/unanswerable_questions.json"
)

report_file = Path(
    "data/evaluation/results/"
    "confidence_score_analysis.json"
)


# --------------------------------
# 2. Load answerable questions
# --------------------------------

answerable_cases = load_evaluation_cases(
    answerable_file
)


# --------------------------------
# 3. Load unanswerable questions
# --------------------------------

with unanswerable_file.open(
    "r",
    encoding="utf-8",
) as file:
    raw_unanswerable = json.load(file)


if not isinstance(
    raw_unanswerable,
    list,
):
    raise ValueError(
        "Unanswerable question file must "
        "contain a JSON list."
    )


unanswerable_cases: list[
    dict[str, str]
] = []


for item in raw_unanswerable:
    if not isinstance(
        item,
        dict,
    ):
        raise ValueError(
            "Each unanswerable case must "
            "be a JSON object."
        )

    case_id = item.get(
        "case_id"
    )

    question = item.get(
        "question"
    )

    if not isinstance(
        case_id,
        str,
    ) or not case_id.strip():
        raise ValueError(
            "Each unanswerable case must "
            "have a valid case_id."
        )

    if not isinstance(
        question,
        str,
    ) or not question.strip():
        raise ValueError(
            f"Case '{case_id}' must have "
            "a valid question."
        )

    unanswerable_cases.append(
        {
            "case_id": case_id,
            "question": question,
        }
    )


# --------------------------------
# 4. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 5. Chroma corpus
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=CHROMA_INDEX_DIR,
    collection_name=DEFAULT_CHROMA_COLLECTION,
)


chunks = vector_store.get_all_chunks()


print(
    "Chroma records:",
    vector_store.count(),
)

print(
    "Corpus chunks:",
    len(chunks),
)


# --------------------------------
# 6. Retrieval stack
# --------------------------------

dense_retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


bm25_retriever = BM25Retriever(
    chunks=chunks,
)


hybrid_retriever = RRFHybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,
    rrf_k=60,
    candidate_pool_size=10,
)


reranker = CrossEncoderReranker(
    base_retriever=hybrid_retriever,
    model_name=(
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    ),
    candidate_pool_size=20,
    device="cpu",
)


# --------------------------------
# 7. Helper
# --------------------------------

def analyze_question(
    case_id: str,
    question: str,
    should_answer: bool,
) -> dict[str, Any]:
    results = reranker.retrieve(
        query=question,
        top_k=5,
    )

    if not results:
        return {
            "case_id": case_id,
            "question": question,
            "should_answer": should_answer,
            "top_score": None,
            "second_score": None,
            "score_margin": None,
            "top_chunk_id": None,
            "top_page_start": None,
            "top_page_end": None,
        }

    top = results[0]

    second = (
        results[1]
        if len(results) >= 2
        else None
    )

    top_score = (
        float(top.score)
        if top.score is not None
        else None
    )

    second_score = (
        float(second.score)
        if (
            second is not None
            and second.score is not None
        )
        else None
    )

    score_margin = None

    if (
        top_score is not None
        and second_score is not None
    ):
        score_margin = (
            top_score
            - second_score
        )

    return {
        "case_id": case_id,
        "question": question,
        "should_answer": should_answer,
        "top_score": top_score,
        "second_score": second_score,
        "score_margin": score_margin,
        "top_chunk_id": top.chunk_id,
        "top_page_start": top.page_start,
        "top_page_end": top.page_end,
    }


# --------------------------------
# 8. Evaluate answerable questions
# --------------------------------

results: list[
    dict[str, Any]
] = []


print(
    "\nAnswerable questions:",
    len(answerable_cases),
)


for index, case in enumerate(
    answerable_cases,
    start=1,
):
    print(
        f"[A {index}/{len(answerable_cases)}] "
        f"{case.case_id}"
    )

    result = analyze_question(
        case_id=case.case_id,
        question=case.question,
        should_answer=True,
    )

    results.append(
        result
    )

    print(
        "    top_score:",
        (
            round(
                result["top_score"],
                4,
            )
            if result["top_score"]
            is not None
            else None
        ),
    )

    print(
        "    margin:",
        (
            round(
                result["score_margin"],
                4,
            )
            if result["score_margin"]
            is not None
            else None
        ),
    )


# --------------------------------
# 9. Evaluate unanswerable questions
# --------------------------------

print(
    "\nUnanswerable questions:",
    len(unanswerable_cases),
)


for index, case in enumerate(
    unanswerable_cases,
    start=1,
):
    print(
        f"[U {index}/{len(unanswerable_cases)}] "
        f"{case['case_id']}"
    )

    result = analyze_question(
        case_id=case["case_id"],
        question=case["question"],
        should_answer=False,
    )

    results.append(
        result
    )

    print(
        "    top_score:",
        (
            round(
                result["top_score"],
                4,
            )
            if result["top_score"]
            is not None
            else None
        ),
    )

    print(
        "    margin:",
        (
            round(
                result["score_margin"],
                4,
            )
            if result["score_margin"]
            is not None
            else None
        ),
    )


# --------------------------------
# 10. Statistics
# --------------------------------

answerable_scores = [
    float(
        result["top_score"]
    )
    for result in results
    if (
        result["should_answer"]
        and result["top_score"]
        is not None
    )
]


unanswerable_scores = [
    float(
        result["top_score"]
    )
    for result in results
    if (
        not result["should_answer"]
        and result["top_score"]
        is not None
    )
]


def summarize(
    values: list[float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(
            values
        ),
        "median": statistics.median(
            values
        ),
        "max": max(values),
    }


answerable_summary = summarize(
    answerable_scores
)

unanswerable_summary = summarize(
    unanswerable_scores
)


# --------------------------------
# 11. Display summary
# --------------------------------

print(
    "\nConfidence Score Analysis"
)

print(
    "-------------------------"
)


print("\nANSWERABLE")

for key, value in (
    answerable_summary.items()
):
    if isinstance(
        value,
        float,
    ):
        print(
            f"{key}:",
            round(
                value,
                4,
            ),
        )
    else:
        print(
            f"{key}:",
            value,
        )


print("\nUNANSWERABLE")

for key, value in (
    unanswerable_summary.items()
):
    if isinstance(
        value,
        float,
    ):
        print(
            f"{key}:",
            round(
                value,
                4,
            ),
        )
    else:
        print(
            f"{key}:",
            value,
        )


# --------------------------------
# 12. Save report
# --------------------------------

report = {
    "retrieval_configuration": {
        "rrf_k": 60,
        "rrf_candidate_pool_size": 10,
        "reranker_model": (
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        "reranker_candidate_pool_size": 20,
        "top_k": 5,
    },

    "answerable_summary": (
        answerable_summary
    ),

    "unanswerable_summary": (
        unanswerable_summary
    ),

    "cases": results,
}


report_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)


report_file.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "\nReport saved to:",
    report_file,
)