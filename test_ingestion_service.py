from pathlib import Path

from docqna.ingestion import (
    DocumentIngestionService,
)


SAMPLE_PDF = Path(
    "data/samples/sample.pdf"
)


def main() -> None:
    service = DocumentIngestionService()

    result = service.process(
        SAMPLE_PDF
    )

    document = result.document
    chunks = result.chunks

    assert document.filename == "sample.pdf"

    assert chunks

    assert all(
        chunk.metadata.get(
            "chunking_strategy"
        )
        == "fixed"
        for chunk in chunks
    )

    assert all(
        chunk.metadata.get(
            "chunk_size"
        )
        == 200
        for chunk in chunks
    )

    assert all(
        chunk.metadata.get(
            "overlap"
        )
        == 40
        for chunk in chunks
    )

    print(
        "Document:",
        document.filename,
    )

    print(
        "Segments:",
        document.segment_count,
    )

    print(
        "Chunks:",
        len(chunks),
    )

    print(
        "Strategy:",
        chunks[0].metadata[
            "chunking_strategy"
        ],
    )

    print(
        "Chunk size:",
        chunks[0].metadata[
            "chunk_size"
        ],
    )

    print(
        "Overlap:",
        chunks[0].metadata[
            "overlap"
        ],
    )

    print()
    print(
        "Document ingestion service test passed."
    )


if __name__ == "__main__":
    main()