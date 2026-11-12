import pytest

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.ingestion.models import DocumentChunk
from experiments.run_phase22b_chunking_experiment import (
    StrategySpec,
    build_chunker,
    chunk_statistics,
    strategy_specs,
)


class FakeEmbeddingModel(BaseEmbeddingModel):
    """Minimal embedding implementation for configuration tests."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_phase22b_chunking_experiment() -> None:
    specs = strategy_specs()

    assert [spec.name for spec in specs] == ["fixed", "structure_aware", "semantic"]

    assert all(spec.chunk_size == 200 for spec in specs)

    assert all(spec.overlap == 40 for spec in specs)

    semantic = specs[2]

    assert semantic.min_chunk_size == 80
    assert semantic.similarity_threshold == 0.45

    embedding_model = FakeEmbeddingModel()

    for spec in specs:
        chunker = build_chunker(spec=spec, embedding_model=embedding_model)

        assert chunker is not None

    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            chunk_index=0,
            text="one two three four",
            filename="sample.txt",
            file_type="txt",
            source_segment_indices=[0],
            metadata={"chunking_strategy": "fixed", "word_count": 4, "sections": []},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            chunk_index=1,
            text="one two",
            filename="sample.txt",
            file_type="txt",
            source_segment_indices=[0],
            metadata={"chunking_strategy": "fixed", "word_count": 2, "sections": []},
        ),
    ]

    stats = chunk_statistics(chunks)

    assert stats.chunk_count == 2
    assert stats.average_words == 3.0
    assert stats.min_words == 2
    assert stats.max_words == 4

    with pytest.raises(ValueError):
        build_chunker(
            spec=StrategySpec(
                name="semantic",
                chunk_size=200,
                overlap=40,
                min_chunk_size=None,
                similarity_threshold=0.45,
            ),
            embedding_model=embedding_model,
        )
