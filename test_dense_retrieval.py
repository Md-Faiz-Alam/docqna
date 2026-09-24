from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.indexing import ChromaVectorStore
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
# 2. Existing persistent index
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
# 4. Search
# --------------------------------

query = (
    "What techniques are used "
    "for identifying project risks?"
)

results = retriever.retrieve(
    query=query,
    top_k=5,
)


print("\nQuery:")
print(query)

print(
    "\nResults:",
    len(results),
)


for result in results:
    print("\n------------------------")
    print("Rank:", result.rank)
    print("Chunk ID:", result.chunk_id)
    print("Distance:", result.distance)
    print("File:", result.filename)
    print(
        "Pages:",
        result.page_start,
        "-",
        result.page_end,
    )
    print(
        "Source segments:",
        result.source_segment_indices,
    )
    print("Section:", result.section)
    print("Text:", result.text[:500])

assert len(results) == 5

assert results[0].rank == 1

assert all(
    result.filename == "sample.pdf"
    for result in results
)

assert all(
    results[index].distance
    <= results[index + 1].distance
    for index in range(
        len(results) - 1
    )
)

print("\nDense retrieval tests passed")

# --------------------------------
# 5. Validation tests
# --------------------------------

try:
    retriever.retrieve(
        query="",
        top_k=5,
    )
except ValueError as exc:
    print(
        "\nEmpty query validation passed:",
        exc,
    )
else:
    raise AssertionError(
        "Empty query should have raised ValueError."
    )


try:
    retriever.retrieve(
        query="project risks",
        top_k=0,
    )
except ValueError as exc:
    print(
        "top_k validation passed:",
        exc,
    )
else:
    raise AssertionError(
        "top_k=0 should have raised ValueError."
    )

# --------------------------------
# 6. Additional semantic query
# --------------------------------

second_query = (
    "How should project risks "
    "be analyzed and managed?"
)

second_results = retriever.retrieve(
    query=second_query,
    top_k=5,
)

print("\nSecond Query:")
print(second_query)

for result in second_results:
    print("\n------------------------")
    print("Rank:", result.rank)
    print("Distance:", result.distance)
    print("Pages:", result.page_start, "-", result.page_end)
    print("Text:", result.text[:400])


assert len(second_results) == 5

print("\nSecond semantic query test passed")