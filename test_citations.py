import re

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.generation.providers.ollama import OllamaProvider
from docqna.indexing import ChromaVectorStore
from docqna.rag import RAGPipeline
from docqna.retrieval import DenseRetriever
from docqna.settings import get_settings


settings = get_settings()


# --------------------------------
# 1. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 2. Existing vector store
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
# 3. Retriever
# --------------------------------

retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


# --------------------------------
# 4. Ollama
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
# 5. RAG pipeline
# --------------------------------

rag = RAGPipeline(
    retriever=retriever,
    llm=llm,
)


# --------------------------------
# 6. Cited question
# --------------------------------

query = (
    "What techniques are used "
    "for identifying project risks?"
)


print("\nQuestion:")
print(query)

print("\nGenerating cited answer...")


response = rag.ask_with_citations(
    query=query,
    top_k=5,
)


# --------------------------------
# 7. Answer
# --------------------------------

print("\nAnswer:")
print(response.answer)


print(
    "\nCited source IDs:",
    response.cited_source_ids,
)


# --------------------------------
# 8. Display cited sources
# --------------------------------

print("\nCited Sources:")


for source in response.sources:
    if (
        source.citation_id
        not in response.cited_source_ids
    ):
        continue

    print("\n------------------------")

    print(
        "Citation:",
        f"[{source.citation_id}]",
    )

    print(
        "File:",
        source.filename,
    )

    print(
        "Pages:",
        source.page_start,
        "-",
        source.page_end,
    )

    print(
        "Chunk ID:",
        source.chunk_id,
    )

    print(
        "Distance:",
        source.distance,
    )

    print(
        "Text:",
        source.text[:500],
    )


# --------------------------------
# 9. Basic validation
# --------------------------------

assert response.answer.strip()

assert response.sources

assert response.cited_source_ids


for citation_id in response.cited_source_ids:
    assert (
        1
        <= citation_id
        <= len(response.sources)
    )

    assert (
        f"[{citation_id}]"
        in response.answer
    )


# --------------------------------
# 10. Citation format validation
# --------------------------------

# Old model-generated citation formats should
# have been normalized by RAGPipeline.

assert re.search(
    r"\[\s*source\b",
    response.answer,
    flags=re.IGNORECASE,
) is None


assert re.search(
    r"\bsource\s*\[\d+\]",
    response.answer,
    flags=re.IGNORECASE,
) is None


# Extract canonical citations such as:
# [1]
# [2]
# [3]

answer_citation_ids = [
    int(value)
    for value in re.findall(
        r"\[(\d+)\]",
        response.answer,
    )
]


assert answer_citation_ids


# --------------------------------
# 11. Citation consistency
# --------------------------------

assert set(answer_citation_ids) == set(
    response.cited_source_ids
)


# Every citation found in the answer
# must refer to an available source.

for citation_id in answer_citation_ids:
    assert (
        1
        <= citation_id
        <= len(response.sources)
    )


print(
    "\nCitation format validation passed"
)

print(
    "Citation test passed"
)