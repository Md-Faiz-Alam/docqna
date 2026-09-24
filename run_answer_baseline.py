import time
from pathlib import Path

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation import (
    AnswerEvaluator,
    load_evaluation_cases,
)
from docqna.generation.providers.ollama import (
    OllamaProvider,
)
from docqna.indexing import ChromaVectorStore
from docqna.rag import RAGPipeline
from docqna.retrieval import DenseRetriever
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
    "dense_rag_answer_baseline.json"
)

checkpoint_file = Path(
    "data/evaluation/results/"
    "dense_rag_answer_baseline.checkpoint.json"
)


# --------------------------------
# 2. Start timer
# --------------------------------

run_started = time.perf_counter()


# --------------------------------
# 3. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 4. Vector store
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=CHROMA_INDEX_DIR,
    collection_name=DEFAULT_CHROMA_COLLECTION,
)


print(
    "Indexed records:",
    vector_store.count(),
)


# --------------------------------
# 5. Retriever
# --------------------------------

retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


# --------------------------------
# 6. Ollama
# --------------------------------

llm = OllamaProvider(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
)


if not llm.health_check():
    raise RuntimeError(
        "Ollama is not reachable."
    )


print("Ollama connection: OK")

print(
    "Model:",
    settings.ollama_model,
)


# --------------------------------
# 7. RAG pipeline
# --------------------------------

rag = RAGPipeline(
    retriever=retriever,
    llm=llm,
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


if checkpoint_file.exists():
    print(
        "Checkpoint found:",
        checkpoint_file,
    )

    print(
        "Completed successful cases "
        "will be resumed."
    )
else:
    print(
        "No checkpoint found. "
        "Starting fresh."
    )


# --------------------------------
# 9. Evaluate
# --------------------------------

evaluator = AnswerEvaluator(
    rag_pipeline=rag
)


report = evaluator.evaluate(
    cases=cases,
    top_k=5,
    checkpoint_path=checkpoint_file,
    resume=True,
)


run_elapsed = (
    time.perf_counter()
    - run_started
)


# --------------------------------
# 10. Aggregate metrics
# --------------------------------

print("\nAnswer Evaluation Baseline")
print("--------------------------")

print(
    "Cases:",
    report.num_cases,
)

print(
    "Successful:",
    report.successful_cases,
)

print(
    "Failed:",
    report.failed_cases,
)

print(
    "Answer Term Coverage:",
    round(
        report.mean_answer_term_coverage,
        4,
    ),
)

print(
    "Citation Presence:",
    round(
        report.mean_citation_presence,
        4,
    ),
)

print(
    "Citation Validity:",
    round(
        report.mean_citation_validity,
        4,
    ),
)

print(
    "Cited Evidence Term Coverage:",
    round(
        report.mean_cited_evidence_term_coverage,
        4,
    ),
)

print(
    "Mean Answer Word Count:",
    round(
        report.mean_answer_word_count,
        2,
    ),
)

print(
    "Mean Case Time:",
    round(
        report.mean_case_elapsed_seconds,
        2,
    ),
    "seconds",
)

print(
    "Total Case Time:",
    round(
        report.total_case_elapsed_seconds,
        2,
    ),
    "seconds",
)

print(
    "Current Run Wall Time:",
    round(
        run_elapsed,
        2,
    ),
    "seconds",
)


# --------------------------------
# 11. Per-case diagnostics
# --------------------------------

for case in report.cases:
    print("\n------------------------")

    print(
        "Case:",
        case.case_id,
    )

    print(
        "Answer coverage:",
        round(
            case.answer_term_coverage,
            4,
        ),
    )

    print(
        "Citation presence:",
        case.citation_presence,
    )

    print(
        "Citation validity:",
        case.citation_validity,
    )

    print(
        "Evidence coverage:",
        round(
            case.cited_evidence_term_coverage,
            4,
        ),
    )

    print(
        "Cited IDs:",
        case.cited_source_ids,
    )

    print(
        "Matched answer terms:",
        case.matched_answer_terms,
    )

    print(
        "Elapsed:",
        round(
            case.elapsed_seconds,
            2,
        ),
        "seconds",
    )

    if case.error is not None:
        print(
            "ERROR:",
            case.error,
        )


# --------------------------------
# 12. Save final report
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


# --------------------------------
# 13. Checkpoint cleanup
# --------------------------------

if report.failed_cases == 0:
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(
        "Evaluation completed successfully."
    )

    print(
        "Checkpoint cleared."
    )

else:
    print(
        "Evaluation contains failed cases."
    )

    print(
        "Checkpoint retained for retry:",
        checkpoint_file,
    )