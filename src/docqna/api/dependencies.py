"""Production object graph for the API.

Heavy objects (embedding model, cross-encoder, Chroma client) are created
lazily and cached for the life of the process. The RAG pipeline itself is
cheap to rebuild and is invalidated whenever the indexed corpus changes.
"""

from collections.abc import Callable
from functools import lru_cache

from sentence_transformers import CrossEncoder

from docqna.embeddings import BaseEmbeddingModel, SentenceTransformerEmbedding
from docqna.exceptions import IndexingError, LLMConnectionError, RetrievalError
from docqna.generation.providers.base import BaseLLMProvider
from docqna.generation.providers.ollama import OllamaProvider
from docqna.indexing import ChromaVectorStore, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRegistry
from docqna.rag import RAGPipeline
from docqna.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RetrievalConfidenceGate,
    RRFHybridRetriever,
)
from docqna.settings import Settings, get_settings

PRODUCTION_CHUNKING_STRATEGY = "fixed"


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
def get_cross_encoder_model() -> CrossEncoder:
    """Return the shared production cross-encoder model."""

    settings = get_settings()
    try:
        return CrossEncoder(settings.reranker_model, device=settings.reranker_device)
    except Exception as exc:
        raise RetrievalError(
            f"Could not load cross-encoder model '{settings.reranker_model}'."
        ) from exc


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return the shared production Chroma vector store."""

    settings = get_settings()
    return ChromaVectorStore(
        persist_directory=settings.chroma_dir, collection_name=settings.chroma_collection
    )


@lru_cache(maxsize=1)
def get_document_registry() -> DocumentRegistry:
    """Return the persistent production document registry."""

    return DocumentRegistry(get_settings().registry_path)


@lru_cache(maxsize=1)
def get_document_ingestion_service() -> DocumentIngestionService:
    """Return the production document processing service."""

    return DocumentIngestionService()


@lru_cache(maxsize=1)
def get_document_index_service() -> DocumentIndexService:
    """Build and cache the production document index service."""

    settings = get_settings()
    return DocumentIndexService(
        ingestion_service=get_document_ingestion_service(),
        registry=get_document_registry(),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
        max_documents_per_session=settings.max_documents_per_session,
        max_active_documents=settings.max_active_documents,
        max_chunks_per_document=settings.max_chunks_per_document,
        document_retention_hours=settings.document_retention_hours,
    )


def build_llm(settings: Settings) -> OllamaProvider:
    """Create the configured LLM provider."""

    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def build_rag_pipeline(
    *,
    settings: Settings,
    embedding_model: BaseEmbeddingModel,
    vector_store: ChromaVectorStore,
    cross_encoder_model: CrossEncoder | None,
    llm: BaseLLMProvider,
) -> RAGPipeline:
    """Assemble dense + BM25 -> RRF -> cross-encoder -> gated generation.

    Raises IndexingError if the corpus is empty or was indexed with a
    chunking strategy other than the evaluated production one.
    """

    record_count = vector_store.count()
    if record_count <= 0:
        raise IndexingError("Production Chroma index is empty.")

    chunks = vector_store.get_all_chunks()
    if len(chunks) != record_count:
        raise IndexingError("Production Chroma chunk count does not match the stored record count.")

    for chunk in chunks:
        strategy = chunk.metadata.get("chunking_strategy")
        if strategy != PRODUCTION_CHUNKING_STRATEGY:
            raise IndexingError(
                f"Production API expected {PRODUCTION_CHUNKING_STRATEGY} chunks "
                f"but found strategy: {strategy!r}."
            )

    hybrid_retriever = RRFHybridRetriever(
        dense_retriever=DenseRetriever(embedding_model=embedding_model, vector_store=vector_store),
        sparse_retriever=BM25Retriever(chunks=chunks),
        rrf_k=settings.rrf_k,
        candidate_pool_size=settings.hybrid_candidate_pool_size,
    )

    reranker = CrossEncoderReranker(
        base_retriever=hybrid_retriever,
        model_name=settings.reranker_model,
        candidate_pool_size=settings.reranker_candidate_pool_size,
        device=settings.reranker_device,
        model=cross_encoder_model,
    )

    return RAGPipeline(
        retriever=reranker,
        llm=llm,
        confidence_gate=RetrievalConfidenceGate(min_top_score=settings.answerability_min_score),
    )


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Build and cache the production RAG pipeline on first QA request."""

    settings = get_settings()
    vector_store = get_vector_store()

    # Fail fast on an empty corpus before loading the cross-encoder.
    if vector_store.count() <= 0:
        raise IndexingError("Production Chroma index is empty.")

    llm = build_llm(settings)
    if not llm.health_check():
        raise LLMConnectionError("Ollama is not reachable.")

    return build_rag_pipeline(
        settings=settings,
        embedding_model=get_embedding_model(),
        vector_store=vector_store,
        cross_encoder_model=get_cross_encoder_model(),
        llm=llm,
    )


def clear_rag_pipeline_cache() -> None:
    """Invalidate the cached RAG stack after document index mutations.

    Model weights and the Chroma client stay cached; the next QA request
    rebuilds BM25/RRF/reranking against the current corpus.
    """

    get_rag_pipeline.cache_clear()


def get_document_index_service_provider() -> Callable[[], DocumentIndexService]:
    """Return the index-service factory without loading models during DI."""

    return get_document_index_service


def get_rag_pipeline_provider() -> Callable[[], RAGPipeline]:
    """Return the RAG pipeline factory without building it during DI.

    FastAPI resolves dependencies before body validation; returning the
    factory keeps invalid requests from triggering model loading.
    """

    return get_rag_pipeline
