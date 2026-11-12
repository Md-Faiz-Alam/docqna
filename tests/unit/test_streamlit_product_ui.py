from docqna.ui.streamlit_app import (
    SUPPORTED_UPLOAD_TYPES,
    build_multipart_file_payload,
    build_qa_payload,
    format_document_label,
)


def test_streamlit_product_ui() -> None:
    assert SUPPORTED_UPLOAD_TYPES == ("pdf", "docx", "txt")

    selected_payload = build_qa_payload(
        question="What is the critical path?", top_k=5, document_id="doc_123"
    )

    assert selected_payload == {
        "question": ("What is the critical path?"),
        "top_k": 5,
        "filters": {"document_id": "doc_123"},
    }

    all_documents_payload = build_qa_payload(
        question="What is project risk?", top_k=5, document_id=None
    )

    assert all_documents_payload == {"question": ("What is project risk?"), "top_k": 5}

    body, content_type = build_multipart_file_payload(
        filename="example.pdf",
        content=b"PDF_BYTES",
        content_type=("application/pdf"),
        boundary="TEST_BOUNDARY",
    )

    assert "boundary=TEST_BOUNDARY" in content_type

    assert b'name="file"; filename="example.pdf"' in body

    assert b"PDF_BYTES" in body

    assert body.endswith(b"--TEST_BOUNDARY--\r\n")

    label = format_document_label({"filename": "sample.pdf", "version": 2, "file_type": "pdf"})

    assert label == ("sample.pdf · v2 · PDF")


def test_sse_parser_reads_token_and_final_events() -> None:
    from docqna.ui.streamlit_app import iter_sse_events

    stream = [
        b"event: token\n",
        b'data: {"text": "Hel"}\n',
        b"\n",
        b"event: final\n",
        b'data: {"answer": "Hello [1].", "abstained": false}\n',
        b"\n",
    ]

    assert list(iter_sse_events(stream)) == [
        ("token", {"text": "Hel"}),
        ("final", {"answer": "Hello [1].", "abstained": False}),
    ]
