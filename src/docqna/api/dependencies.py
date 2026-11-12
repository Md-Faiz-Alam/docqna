"""Production object graph for the API.

Heavy objects (embedding model, cross-encoder, Chroma client, RAG
pipeline) are created lazily and cached for the life of the process.
Document mutations invalidate only the affected session's BM25 index.
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
from docqna.ingestion.models import DocumentChunk
from docqna.rag import RAGPipeline
from docqna.rag.grounding import GroundingVerifier
from docqna.retrieval import CrossEncoderReranker, DenseRetriever, RetrievalConfidenceGate
from docqna.retrieval.session_index import ChunkLoader, SessionScopedHybridRetriever
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


def make_chunk_loader(vector_store: ChromaVectorStore) -> ChunkLoader:
    """Load one session's chunks, enforcing the evaluated chunking strategy."""

    def load(session_id: str | None) -> list[DocumentChunk]:
        chunks = vector_store.get_all_chunks(session_id=session_id)
        for chunk in chunks:
            strategy = chunk.metadata.get("chunking_strategy")
            if strategy != PRODUCTION_CHUNKING_STRATEGY:
                raise IndexingError(
                    f"Production API expected {PRODUCTION_CHUNKING_STRATEGY} chunks "
                    f"but found strategy: {strategy!r}."
                )
        return chunks

    return load


def build_session_retriever(
    *, settings: Settings, embedding_model: BaseEmbeddingModel, vector_store: ChromaVectorStore
) -> SessionScopedHybridRetriever:
    return SessionScopedHybridRetriever(
        dense_retriever=DenseRetriever(embedding_model=embedding_model, vector_store=vector_store),
        load_chunks=make_chunk_loader(vector_store),
        rrf_k=settings.rrf_k,
        candidate_pool_size=settings.hybrid_candidate_pool_size,
        max_cached_sessions=settings.max_cached_session_indexes,
    )


def build_rag_pipeline(
    *,
    settings: Settings,
    session_retriever: SessionScopedHybridRetriever,
    cross_encoder_model: CrossEncoder | None,
    llm: BaseLLMProvider,
) -> RAGPipeline:
    """Assemble dense + per-session BM25 -> RRF -> cross-encoder -> gated generation."""

    reranker = CrossEncoderReranker(
        base_retriever=session_retriever,
        model_name=settings.reranker_model,
        candidate_pool_size=settings.reranker_candidate_pool_size,
        device=settings.reranker_device,
        model=cross_encoder_model,
    )

    grounding_verifier = (
        GroundingVerifier(scorer=reranker.model, min_score=settings.grounding_min_score)
        if settings.grounding_check_enabled
        else None
    )

    return RAGPipeline(
        retriever=reranker,
        llm=llm,
        confidence_gate=RetrievalConfidenceGate(min_top_score=settings.answerability_min_score),
        grounding_verifier=grounding_verifier,
    )


@lru_cache(maxsize=1)
def get_session_retriever() -> SessionScopedHybridRetriever:
    """Shared retriever holding the per-session BM25 index cache."""

    return build_session_retriever(
        settings=get_settings(),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_llm() -> BaseLLMProvider:
    return build_llm(get_settings())


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Build the production RAG pipeline once, on the first QA request.

    The pipeline is never rebuilt on uploads/deletes; only the affected
    session's BM25 index is invalidated (see invalidate_session_index).
    """

    llm = get_llm()
    if not llm.health_check():
        raise LLMConnectionError("Ollama is not reachable.")

    return build_rag_pipeline(
        settings=get_settings(),
        session_retriever=get_session_retriever(),
        cross_encoder_model=get_cross_encoder_model(),
        llm=llm,
    )


def invalidate_session_index(session_id: str | None = None) -> None:
    """Drop cached BM25 state after a document mutation.

    Pass the mutated session, or None to drop every session (retention purge).
    Does nothing if no QA request has built the retriever yet.
    """

    if get_session_retriever.cache_info().currsize:
        get_session_retriever().invalidate(session_id)


def get_document_index_service_provider() -> Callable[[], DocumentIndexService]:
    """Return the index-service factory without loading models during DI."""

    return get_document_index_service


def get_rag_pipeline_provider() -> Callable[[], RAGPipeline]:
    """Return the RAG pipeline factory without building it during DI.

    FastAPI resolves dependencies before body validation; returning the
    factory keeps invalid requests from triggering model loading.
    """

    return get_rag_pipeline
