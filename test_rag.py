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
# 2. Existing Chroma index
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
# 3. Dense retriever
# --------------------------------

retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


# --------------------------------
# 4. Ollama provider
# --------------------------------

llm = OllamaProvider(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
)


# --------------------------------
# 5. Ollama health check
# --------------------------------

if not llm.health_check():
    raise RuntimeError(
        "Ollama is not reachable. "
        "Make sure Ollama is running."
    )

print(
    "Ollama connection:",
    "OK",
)

print(
    "Model:",
    settings.ollama_model,
)


# --------------------------------
# 6. RAG pipeline
# --------------------------------

rag = RAGPipeline(
    retriever=retriever,
    llm=llm,
)


# --------------------------------
# 7. Ask question
# --------------------------------

query = (
    "What techniques are used "
    "for identifying project risks?"
)

print("\nQuestion:")
print(query)

print("\nGenerating answer...")


answer = rag.ask(
    query=query,
    top_k=5,
)


# --------------------------------
# 8. Display answer
# --------------------------------

print("\nAnswer:")
print(answer)


# --------------------------------
# 9. Basic validation
# --------------------------------

assert answer.strip()

print("\nBasic RAG test passed")