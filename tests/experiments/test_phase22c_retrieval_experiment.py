from typing import cast

import pytest

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.indexing.base import BaseVectorStore
from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import BM25Retriever, DenseRetriever, RRFHybridRetriever
from experiments.run_phase22c_retrieval_experiment import (
    RetrieverName,
    RetrieverSpec,
    build_retriever,
    retriever_specs,
)


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Minimal embedding model for configuration tests."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def build_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            chunk_index=0,
            text="one two three",
            filename="sample.txt",
            file_type="txt",
            source_segment_indices=[0],
            metadata={"chunking_strategy": "fixed", "word_count": 3, "sections": []},
        )
    ]


def test_phase22c_retrieval_experiment() -> None:
    specs = retriever_specs()

    assert [spec.name for spec in specs] == ["dense", "bm25", "rrf", "rrf_reranker"]

    embedding_model = FakeEmbeddingModel()

    # The lightweight test only exercises constructors. DenseRetriever
    # stores the vector-store dependency without querying it here.
    vector_store = cast(BaseVectorStore, object())

    chunks = build_chunks()

    dense = build_retriever(
        spec=specs[0], chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    assert isinstance(dense, DenseRetriever)

    bm25 = build_retriever(
        spec=specs[1], chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    assert isinstance(bm25, BM25Retriever)

    rrf = build_retriever(
        spec=specs[2], chunks=chunks, embedding_model=embedding_model, vector_store=vector_store
    )

    assert isinstance(rrf, RRFHybridRetriever)

    # Avoid loading the real CrossEncoder in this lightweight unit
    # test. The exact reranker class/configuration is exercised by
    # the real experiment runner.
    reranker_spec = specs[3]

    assert reranker_spec.name == "rrf_reranker"

    assert "CrossEncoder" in (reranker_spec.description)

    invalid_name = cast(RetrieverName, "unknown")

    with pytest.raises(ValueError):
        build_retriever(
            spec=RetrieverSpec(name=invalid_name, description="invalid"),
            chunks=chunks,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
