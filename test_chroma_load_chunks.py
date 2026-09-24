from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.indexing import ChromaVectorStore


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
    "Reconstructed chunks:",
    len(chunks),
)


assert chunks

assert (
    len(chunks)
    == vector_store.count()
)

assert len(chunks) == 192


first = chunks[0]

print("\nFirst chunk:")
print("ID:", first.chunk_id)
print("Index:", first.chunk_index)
print("File:", first.filename)
print(
    "Pages:",
    first.page_start,
    "-",
    first.page_end,
)

print(
    "Text:",
    first.text[:200],
)


print(
    "\nChroma chunk reconstruction test passed"
)