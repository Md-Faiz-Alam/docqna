from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest
from sentence_transformers import CrossEncoder

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.generation.providers.base import BaseLLMProvider
from docqna.indexing import ChromaVectorStore
from docqna.ingestion.models import DocumentChunk
from docqna.rag import RAGPipeline
from docqna.retrieval import BM25Retriever, CrossEncoderReranker, DenseRetriever, RRFHybridRetriever


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Deterministic embedding model for retrieval tests."""

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lower = text.lower()

        return [1.0, float(lower.count("finance")), float(lower.count("schedule"))]


class FakeCrossEncoder:
    """Small stand-in for CrossEncoder.predict()."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [(2.0 if "finance" in text.lower() else 1.0) for _, text in pairs]


class FakeLLM:
    """Minimal LLM used by RAGPipeline."""

    def generate(self, prompt: str) -> str:
        return "Filtered answer [1]"


def build_chunk(
    *,
    chunk_id: str,
    chunk_index: int,
    text: str,
    filename: str,
    file_type: str,
    document_id: str,
    document_version: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        text=text,
        filename=filename,
        file_type=file_type,
        source_segment_indices=[chunk_index],
        page_start=1,
        page_end=1,
        section=None,
        metadata={
            "chunking_strategy": "fixed",
            "word_count": len(text.split()),
            "sections": [],
            "document_id": document_id,
            "document_version": (document_version),
            "content_hash": (document_id.replace("doc_", "") * 8)[:64],
            "original_chunk_id": chunk_id,
        },
    )


def test_metadata_filters() -> None:
    chunks = [
        build_chunk(
            chunk_id="chunk_a",
            chunk_index=0,
            text=("Project scheduling and milestone planning guidance."),
            filename="alpha.pdf",
            file_type="pdf",
            document_id="doc_alpha",
            document_version=1,
        ),
        build_chunk(
            chunk_id="chunk_b",
            chunk_index=0,
            text=("Project finance risk and cost management guidance."),
            filename="beta.txt",
            file_type="txt",
            document_id="doc_beta",
            document_version=2,
        ),
    ]

    embedding_model = FakeEmbeddingModel()

    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        vector_store = ChromaVectorStore(
            persist_directory=(Path(temp_dir) / "chroma"), collection_name=("metadata_filter_test")
        )

        embeddings = embedding_model.embed_documents([chunk.text for chunk in chunks])

        vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        # ---------------------------------
        # Chroma metadata reconstruction
        # ---------------------------------

        stored_chunks = vector_store.get_all_chunks()

        assert len(stored_chunks) == 2

        by_filename = {chunk.filename: chunk for chunk in stored_chunks}

        assert by_filename["beta.txt"].metadata["document_id"] == "doc_beta"

        assert by_filename["beta.txt"].metadata["document_version"] == 2

        assert by_filename["beta.txt"].metadata["original_chunk_id"] == "chunk_b"

        # ---------------------------------
        # Dense filtering
        # ---------------------------------

        dense = DenseRetriever(embedding_model=embedding_model, vector_store=vector_store)

        dense_filtered = dense.retrieve(
            query="project finance", top_k=5, metadata_filters={"document_id": "doc_beta"}
        )

        assert len(dense_filtered) == 1
        assert dense_filtered[0].document_id == "doc_beta"
        assert dense_filtered[0].filename == "beta.txt"

        # No-filter behavior remains available.
        dense_unfiltered = dense.retrieve(query="project", top_k=5)

        assert len(dense_unfiltered) == 2

        # ---------------------------------
        # BM25 filtering
        # ---------------------------------

        bm25 = BM25Retriever(chunks=stored_chunks)

        sparse_filtered = bm25.retrieve(
            query="project", top_k=5, metadata_filters={"file_type": "TXT", "document_version": 2}
        )

        assert len(sparse_filtered) == 1
        assert sparse_filtered[0].document_id == "doc_beta"

        # ---------------------------------
        # Hybrid filtering
        # ---------------------------------

        hybrid = RRFHybridRetriever(
            dense_retriever=dense, sparse_retriever=bm25, rrf_k=60, candidate_pool_size=10
        )

        hybrid_filtered = hybrid.retrieve(
            query="project finance", top_k=5, metadata_filters={"filename": "beta.txt"}
        )

        assert len(hybrid_filtered) == 1
        assert hybrid_filtered[0].filename == "beta.txt"

        # ---------------------------------
        # Reranker propagation without
        # loading the real cross-encoder
        # ---------------------------------

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)

        reranker.base_retriever = hybrid
        reranker.model_name = "fake"
        reranker.candidate_pool_size = 20
        reranker.model = cast(CrossEncoder, FakeCrossEncoder())

        reranked = reranker.retrieve(
            query="project finance", top_k=5, metadata_filters={"document_id": "doc_beta"}
        )

        assert len(reranked) == 1
        assert reranked[0].document_id == "doc_beta"

        # ---------------------------------
        # RAG pipeline propagation
        # ---------------------------------

        pipeline = RAGPipeline(retriever=reranker, llm=cast(BaseLLMProvider, FakeLLM()))

        response = pipeline.ask_with_citations(
            query="What is the finance guidance?",
            top_k=5,
            metadata_filters={"document_id": "doc_beta"},
        )

        assert response.answer == ("Filtered answer [1]")

        assert len(response.sources) == 1

        assert response.sources[0].filename == "beta.txt"

        # ---------------------------------
        # Filter validation
        # ---------------------------------

        with pytest.raises(ValueError):
            dense.retrieve(query="project", metadata_filters={"unsupported": "value"})
