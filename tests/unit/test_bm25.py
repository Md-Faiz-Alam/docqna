import pytest

from docqna.ingestion.models import DocumentChunk
from docqna.retrieval import BM25Retriever


def test_bm25() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            chunk_index=0,
            text=(
                "Project risk identification uses brainstorming, "
                "checklists, SWOT analysis, and historical data."
            ),
            filename="sample.pdf",
            file_type="pdf",
            source_segment_indices=[0],
            page_start=89,
            page_end=89,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            chunk_index=1,
            text=("Earned Value Management uses Planned Value, Actual Cost, and Earned Value."),
            filename="sample.pdf",
            file_type="pdf",
            source_segment_indices=[1],
            page_start=56,
            page_end=56,
        ),
        DocumentChunk(
            chunk_id="chunk_3",
            chunk_index=2,
            text=("A Gantt chart displays project activities against a timeline."),
            filename="sample.pdf",
            file_type="pdf",
            source_segment_indices=[2],
            page_start=40,
            page_end=40,
        ),
    ]

    retriever = BM25Retriever(chunks=chunks)

    # --------------------------------
    # Risk query
    # --------------------------------

    results = retriever.retrieve(
        query=("What techniques are used for project risk identification?"), top_k=3
    )

    assert results

    assert results[0].chunk_id == ("chunk_1")

    assert results[0].rank == 1

    assert results[0].retrieval_method == "bm25"

    assert results[0].score is not None

    assert results[0].score > 0

    # --------------------------------
    # EVM query
    # --------------------------------

    results = retriever.retrieve(query=("What are Planned Value and Actual Cost?"), top_k=3)

    assert results

    assert results[0].chunk_id == ("chunk_2")

    # --------------------------------
    # Validation
    # --------------------------------

    with pytest.raises(ValueError):
        retriever.retrieve(query="", top_k=5)

    with pytest.raises(ValueError):
        retriever.retrieve(query="risk", top_k=0)
