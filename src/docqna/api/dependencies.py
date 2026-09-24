from collections.abc import Callable
from functools import lru_cache

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.exceptions import (
    IndexingError,
    LLMConnectionError,
)
from docqna.generation.providers.ollama import OllamaProvider
from docqna.indexing import (
    ChromaVectorStore,
    DocumentIndexService,
)
from docqna.ingestion import (
    DocumentIngestionService,
    DocumentRegistry,
)
from docqna.rag import RAGPipeline
from docqna.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RetrievalConfidenceGate,
    RRFHybridRetriever,
)
from docqna.settings import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformerEmbedding:
    """Return the shared production embedding model."""

    settings = get_settings()

    return SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return the shared production Chroma vector store."""

    return ChromaVectorStore(
        persist_directory=CHROMA_INDEX_DIR,
        collection_name=DEFAULT_CHROMA_COLLECTION,
    )


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    """Return the persistent production document registry."""

    return DocumentRegistry()


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    """Return the production document processing service."""

    return DocumentIngestionService()


@lru_cache(maxsize=1)
def get_document_index_service() -> DocumentIndexService:
    """
    Build and cache the production document index service.
    """

    return DocumentIndexService(
        ingestion_service=get_document_ingestion_service(),
        registry=get_document_registry(),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
    )


def get_document_index_service_provider(
) -> Callable[[], DocumentIndexService]:
    """
    Return the document index service factory without
    initializing the embedding model during dependency
    resolution.
    """

    return get_document_index_service


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """
    Build and cache the production DocQ&A RAG pipeline.

    The pipeline is initialized lazily on the first request
    that needs question answering.
    """

    settings = get_settings()

    embedding_model = get_embedding_model()
    vector_store = get_vector_store()

    record_count = vector_store.count()

    if record_count <= 0:
        raise IndexingError(
            "Production Chroma index is empty."
        )

    chunks = vector_store.get_all_chunks()

    if len(chunks) != record_count:
        raise IndexingError(
            "Production Chroma chunk count does not "
            "match the stored record count."
        )

    for chunk in chunks:
        strategy = chunk.metadata.get(
            "chunking_strategy"
        )

        if strategy != "fixed":
            raise IndexingError(
                "Production API expected fixed chunks "
                f"but found strategy: {strategy!r}."
            )

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

    confidence_gate = RetrievalConfidenceGate(
        min_top_score=0.0,
    )

    llm = OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    if not llm.health_check():
        raise LLMConnectionError(
            "Ollama is not reachable."
        )

    return RAGPipeline(
        retriever=reranker,
        llm=llm,
        confidence_gate=confidence_gate,
    )


def clear_rag_pipeline_cache() -> None:
    """
    Invalidate the cached retrieval/RAG stack after
    document index mutations.

    The shared embedding model and Chroma client remain
    cached. The next QA request rebuilds BM25/RRF/reranking
    against the current corpus.
    """

    get_rag_pipeline.cache_clear()


def get_rag_pipeline_provider() -> Callable[[], RAGPipeline]:
    """
    Return the cached RAG pipeline factory without initializing it.

    FastAPI may resolve dependencies before request-body validation.
    Returning the factory itself keeps dependency resolution lightweight;
    the expensive production pipeline is created only when the endpoint
    executes after validation succeeds.
    """

    return get_rag_pipeline
