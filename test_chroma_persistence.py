from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.indexing import ChromaVectorStore


vector_store = ChromaVectorStore(
    persist_directory=CHROMA_INDEX_DIR,
    collection_name=DEFAULT_CHROMA_COLLECTION,
)

count = vector_store.count()

print("Persistent record count:", count)

assert count > 0

print("Persistence test passed")