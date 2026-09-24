from pathlib import Path

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.evaluation import (
    RetrievalEvaluator,
    load_evaluation_cases,
)
from docqna.indexing import ChromaVectorStore
from docqna.retrieval import BM25Retriever


# --------------------------------
# 1. Paths
# --------------------------------

evaluation_file = Path(
    "data/evaluation/baseline_questions.json"
)

report_file = Path(
    "data/evaluation/results/"
    "bm25_retrieval_baseline.json"
)


# --------------------------------
# 2. Existing Chroma corpus
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
    "BM25 corpus chunks:",
    len(chunks),
)


if len(chunks) != vector_store.count():
    raise RuntimeError(
        "BM25 corpus size does not match "
        "the Chroma corpus."
    )


# --------------------------------
# 3. BM25 retriever
# --------------------------------

retriever = BM25Retriever(
    chunks=chunks,
)


# --------------------------------
# 4. Evaluation cases
# --------------------------------

cases = load_evaluation_cases(
    evaluation_file
)


print(
    "Evaluation cases:",
    len(cases),
)


# --------------------------------
# 5. Evaluate
# --------------------------------

evaluator = RetrievalEvaluator(
    retriever=retriever
)


report = evaluator.evaluate(
    cases=cases,
    top_k=5,
)


# --------------------------------
# 6. Aggregate metrics
# --------------------------------

print("\nBM25 Retrieval Baseline")
print("-----------------------")

print(
    "Hit@1:",
    round(
        report.mean_hit_at_1,
        4,
    ),
)

print(
    "Hit@3:",
    round(
        report.mean_hit_at_3,
        4,
    ),
)

print(
    "Hit@5:",
    round(
        report.mean_hit_at_5,
        4,
    ),
)

print(
    "MRR@5:",
    round(
        report.mean_reciprocal_rank,
        4,
    ),
)

print(
    "Page Recall@5:",
    round(
        report.mean_page_recall_at_5,
        4,
    ),
)


# --------------------------------
# 7. Per-case diagnostics
# --------------------------------

for case in report.cases:
    print("\n------------------------")

    print(
        "Case:",
        case.case_id,
    )

    print(
        "Question:",
        case.question,
    )

    print(
        "Retrieved pages:",
        case.retrieved_pages,
    )

    # RetrievalEvaluator stores the shared
    # distance representation. BM25 uses
    # negative score as distance.
    print(
        "BM25 scores:",
        [
            round(
                -value,
                4,
            )
            for value
            in case.distances
        ],
    )


# --------------------------------
# 8. Save report
# --------------------------------

report_file.parent.mkdir(
    parents=True,
    exist_ok=True,
)


report_file.write_text(
    report.model_dump_json(
        indent=2
    ),
    encoding="utf-8",
)


print(
    "\nReport saved to:",
    report_file,
)