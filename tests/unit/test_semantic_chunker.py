from collections.abc import Callable
from pathlib import Path

from docqna.ingestion.chunking.semantic import SemanticChunker
from docqna.ingestion.models import DocumentSegment, LoadedDocument


class FakeEmbeddingModel:
    """Deterministic semantic embeddings for chunker tests."""

    @property
    def dimension(self) -> int:
        return 2

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()

        if any(token in normalized for token in ("cat", "feline", "purr", "mouse")):
            return [1.0, 0.0]

        if any(token in normalized for token in ("database", "sql", "record", "row")):
            return [0.0, 1.0]

        return [0.7, 0.7]


def build_document() -> LoadedDocument:
    return LoadedDocument(
        source_path=Path("semantic_test.pdf"),
        filename="semantic_test.pdf",
        file_type="pdf",
        segments=[
            DocumentSegment(
                text=(
                    "Cats chase small mice. "
                    "Felines purr very softly. "
                    "Databases store durable records. "
                    "SQL retrieves matching rows."
                ),
                segment_index=0,
                page_number=1,
                section="Example",
            )
        ],
    )


def test_semantic_boundary() -> None:
    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        chunk_size=30,
        overlap=0,
        min_chunk_size=4,
        similarity_threshold=0.5,
    )

    chunks = chunker.chunk(build_document())

    assert len(chunks) == 2

    assert "Cats chase" in chunks[0].text
    assert "Felines purr" in chunks[0].text
    assert "Databases store" not in chunks[0].text

    assert "Databases store" in chunks[1].text
    assert "SQL retrieves" in chunks[1].text


def test_metadata_and_ids() -> None:
    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        chunk_size=30,
        overlap=0,
        min_chunk_size=4,
        similarity_threshold=0.5,
    )

    first_run = chunker.chunk(build_document())

    second_run = chunker.chunk(build_document())

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]

    for chunk in first_run:
        assert chunk.metadata["chunking_strategy"] == "semantic"
        assert chunk.metadata["similarity_threshold"] == 0.5
        assert chunk.metadata["embedding_dimension"] == 2
        assert chunk.page_start == 1
        assert chunk.page_end == 1
        assert chunk.section == "Example"


def test_overlap_and_size_limit() -> None:
    document = LoadedDocument(
        source_path=Path("overlap.txt"),
        filename="overlap.txt",
        file_type="txt",
        segments=[
            DocumentSegment(
                text=(
                    "Cats chase mice quickly. Felines purr softly indoors. Cats sleep near windows."
                ),
                segment_index=0,
                page_number=None,
                section=None,
            )
        ],
    )

    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        chunk_size=7,
        overlap=2,
        min_chunk_size=4,
        similarity_threshold=-1.0,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) >= 2

    assert all(len(chunk.text.split()) <= 7 for chunk in chunks)

    # The second chunk receives the two-word tail of the first.
    assert chunks[1].text.startswith("mice quickly")


def test_section_boundary() -> None:
    document = LoadedDocument(
        source_path=Path("sections.docx"),
        filename="sections.docx",
        file_type="docx",
        segments=[
            DocumentSegment(
                text="Cats chase small mice.", segment_index=0, page_number=None, section="Animals"
            ),
            DocumentSegment(
                text="Felines purr very softly.",
                segment_index=1,
                page_number=None,
                section="Databases",
            ),
        ],
    )

    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        chunk_size=30,
        overlap=3,
        min_chunk_size=4,
        similarity_threshold=-1.0,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].section == "Animals"
    assert chunks[1].section == "Databases"

    assert not chunks[1].text.startswith("chase small mice")


def test_invalid_configuration() -> None:
    model = FakeEmbeddingModel()

    invalid_factories: list[Callable[[], SemanticChunker]] = [
        lambda: SemanticChunker(embedding_model=model, chunk_size=0),
        lambda: SemanticChunker(embedding_model=model, chunk_size=10, overlap=10),
        lambda: SemanticChunker(embedding_model=model, chunk_size=10, min_chunk_size=11),
        lambda: SemanticChunker(embedding_model=model, similarity_threshold=1.1),
    ]

    for factory in invalid_factories:
        try:
            factory()

        except ValueError:
            continue

        raise AssertionError("Invalid semantic chunker configuration was not rejected.")
