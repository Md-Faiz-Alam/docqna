from pathlib import Path

from docqna.ingestion.chunking import (
    StructureAwareChunker,
)
from docqna.ingestion.models import (
    DocumentSegment,
    LoadedDocument,
)


document = LoadedDocument(
    source_path=Path(
        "synthetic.pdf"
    ),
    filename="synthetic.pdf",
    file_type="pdf",
    segments=[
        DocumentSegment(
            text=(
                "Project planning begins with clear objectives. "
                "Stakeholders define expected project outcomes. "
                "The project team then develops the initial plan."
            ),
            segment_index=0,
            page_number=1,
            section="Planning",
        ),
        DocumentSegment(
            text=(
                "Schedules identify activities and dependencies. "
                "Duration estimates support realistic timelines. "
                "Resources are assigned to planned activities."
            ),
            segment_index=1,
            page_number=2,
            section="Planning",
        ),
        DocumentSegment(
            text=(
                "Risk identification begins with brainstorming. "
                "Checklists help identify common project risks. "
                "SWOT analysis evaluates internal and external factors."
            ),
            segment_index=2,
            page_number=3,
            section="Risk Management",
        ),
        DocumentSegment(
            text=(
                "Qualitative analysis evaluates probability and impact. "
                "High priority risks require response planning. "
                "Risk owners monitor identified threats."
            ),
            segment_index=3,
            page_number=4,
            section="Risk Management",
        ),
    ],
)


chunker = StructureAwareChunker(
    chunk_size=20,
    overlap=4,
)


chunks = chunker.chunk(
    document
)


print(
    "Generated chunks:",
    len(chunks),
)


for chunk in chunks:
    print(
        "\n------------------------"
    )

    print(
        "Chunk:",
        chunk.chunk_index,
    )

    print(
        "ID:",
        chunk.chunk_id,
    )

    print(
        "Words:",
        chunk.metadata[
            "word_count"
        ],
    )

    print(
        "Pages:",
        chunk.page_start,
        "-",
        chunk.page_end,
    )

    print(
        "Section:",
        chunk.section,
    )

    print(
        "Source segments:",
        chunk.source_segment_indices,
    )

    print(
        "Text:",
        chunk.text,
    )


# --------------------------------
# Basic validation
# --------------------------------

assert chunks

for index, chunk in enumerate(
    chunks
):
    assert (
        chunk.chunk_index
        == index
    )

    assert (
        chunk.metadata[
            "chunking_strategy"
        ]
        == "structure_aware"
    )

    assert (
        chunk.metadata[
            "word_count"
        ]
        <= 20
    )

    assert chunk.text.strip()

    # Explicit section boundaries
    # must never be mixed.
    assert (
        len(
            chunk.metadata[
                "sections"
            ]
        )
        <= 1
    )


# --------------------------------
# Section-boundary validation
# --------------------------------

for chunk in chunks:
    sections = chunk.metadata[
        "sections"
    ]

    if sections:
        assert (
            chunk.section
            == sections[0]
        )


# --------------------------------
# Deterministic ID validation
# --------------------------------

second_run = chunker.chunk(
    document
)


assert [
    chunk.chunk_id
    for chunk in chunks
] == [
    chunk.chunk_id
    for chunk in second_run
]


# --------------------------------
# Invalid configuration
# --------------------------------

try:
    StructureAwareChunker(
        chunk_size=0,
        overlap=0,
    )

    raise AssertionError(
        "chunk_size=0 should fail."
    )

except ValueError:
    pass


try:
    StructureAwareChunker(
        chunk_size=20,
        overlap=20,
    )

    raise AssertionError(
        "overlap >= chunk_size should fail."
    )

except ValueError:
    pass


print(
    "\nStructure-aware chunker test passed"
)