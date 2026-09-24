from pathlib import Path

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation import (
    RetrievalEvaluator,
    load_evaluation_cases,
)
from docqna.indexing import ChromaVectorStore
from docqna.retrieval import (
    BM25Retriever,
    DenseRetriever,
    RRFHybridRetriever,
)
from docqna.settings import get_settings


settings = get_settings()


# --------------------------------
# 1. Paths
# --------------------------------

evaluation_file = Path(
    "data/evaluation/baseline_questions.json"
)

report_file = Path(
    "data/evaluation/results/"
    "rrf_retrieval_baseline.json"
)


# --------------------------------
# 2. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 3. Existing Chroma corpus
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
    "Hybrid corpus chunks:",
    len(chunks),
)


if len(chunks) != vector_store.count():
    raise RuntimeError(
        "Hybrid corpus size does not match "
        "the Chroma corpus."
    )


# --------------------------------
# 4. Dense retriever
# --------------------------------

dense_retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


# --------------------------------
# 5. BM25 retriever
# --------------------------------

bm25_retriever = BM25Retriever(
    chunks=chunks,
)


# --------------------------------
# 6. Hybrid RRF retriever
# --------------------------------

hybrid_retriever = RRFHybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,

    # Standard baseline RRF constant.
    rrf_k=60,

    # Retrieve a broader candidate pool
    # from each retrieval system before
    # fusing them.
    candidate_pool_size=10,
)


# --------------------------------
# 7. Evaluation cases
# --------------------------------

cases = load_evaluation_cases(
    evaluation_file
)


print(
    "Evaluation cases:",
    len(cases),
)


# --------------------------------
# 8. Evaluate
# --------------------------------

evaluator = RetrievalEvaluator(
    retriever=hybrid_retriever
)


report = evaluator.evaluate(
    cases=cases,
    top_k=5,
)


# --------------------------------
# 9. Aggregate metrics
# --------------------------------

print("\nHybrid RRF Retrieval Baseline")
print("-----------------------------")

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
# 10. Per-case diagnostics
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

    print(
        "RRF scores:",
        [
            round(
                -value,
                6,
            )
            for value
            in case.distances
        ],
    )


# --------------------------------
# 11. Save report
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