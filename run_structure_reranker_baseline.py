from pathlib import Path

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation import (
    RetrievalEvaluator,
    load_evaluation_cases,
)
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

STRUCTURE_INDEX_DIR = Path(
    "data/indexes/chroma_structure"
)

STRUCTURE_COLLECTION = (
    "docqna_structure_documents"
)

evaluation_file = Path(
    "data/evaluation/baseline_questions.json"
)

report_file = Path(
    "data/evaluation/results/"
    "structure_reranker_retrieval_baseline.json"
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
# 3. Structure-aware Chroma index
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=STRUCTURE_INDEX_DIR,
    collection_name=STRUCTURE_COLLECTION,
)


chunks = vector_store.get_all_chunks()


print(
    "Structure Chroma records:",
    vector_store.count(),
)

print(
    "Structure corpus chunks:",
    len(chunks),
)


if len(chunks) != vector_store.count():
    raise RuntimeError(
        "Structure corpus size does not match "
        "the Chroma record count."
    )


if not chunks:
    raise RuntimeError(
        "Structure-aware index is empty."
    )


for chunk in chunks:
    if (
        chunk.metadata.get(
            "chunking_strategy"
        )
        != "structure_aware"
    ):
        raise RuntimeError(
            "Unexpected chunking strategy found "
            "in structure-aware index."
        )


print(
    "Chunking strategy validation: OK"
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
# 6. Hybrid RRF
# --------------------------------

hybrid_retriever = RRFHybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,
    rrf_k=60,
    candidate_pool_size=10,
)


# --------------------------------
# 7. Cross-encoder reranker
# --------------------------------

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
# 8. Evaluation cases
# --------------------------------

cases = load_evaluation_cases(
    evaluation_file
)


print(
    "Evaluation cases:",
    len(cases),
)


# --------------------------------
# 9. Evaluate
# --------------------------------

evaluator = RetrievalEvaluator(
    retriever=reranker
)


report = evaluator.evaluate(
    cases=cases,
    top_k=5,
)


# --------------------------------
# 10. Aggregate metrics
# --------------------------------

print(
    "\nStructure-Aware + RRF + Cross-Encoder"
)

print(
    "-------------------------------------"
)


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
# 11. Per-case diagnostics
# --------------------------------

for case in report.cases:
    print(
        "\n------------------------"
    )

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
        "Cross-encoder scores:",
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
# 12. Save report
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