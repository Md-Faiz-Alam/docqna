from docqna.ui.streamlit_app import (
    SUPPORTED_UPLOAD_TYPES,
    build_multipart_file_payload,
    build_qa_payload,
    format_document_label,
)


def main() -> None:
    assert SUPPORTED_UPLOAD_TYPES == (
        "pdf",
        "docx",
        "txt",
    )

    print(
        "Supported upload types: PASS"
    )

    selected_payload = build_qa_payload(
        question="What is the critical path?",
        top_k=5,
        document_id="doc_123",
    )

    assert selected_payload == {
        "question": (
            "What is the critical path?"
        ),
        "top_k": 5,
        "filters": {
            "document_id": "doc_123",
        },
    }

    all_documents_payload = (
        build_qa_payload(
            question="What is project risk?",
            top_k=5,
            document_id=None,
        )
    )

    assert all_documents_payload == {
        "question": (
            "What is project risk?"
        ),
        "top_k": 5,
    }

    print(
        "Document-scoped QA payloads: PASS"
    )

    body, content_type = (
        build_multipart_file_payload(
            filename="example.pdf",
            content=b"PDF_BYTES",
            content_type=(
                "application/pdf"
            ),
            boundary="TEST_BOUNDARY",
        )
    )

    assert (
        "boundary=TEST_BOUNDARY"
        in content_type
    )

    assert (
        b'name="file"; filename="example.pdf"'
        in body
    )

    assert (
        b"PDF_BYTES"
        in body
    )

    assert body.endswith(
        b"--TEST_BOUNDARY--\r\n"
    )

    print(
        "Multipart upload body: PASS"
    )

    label = format_document_label(
        {
            "filename": "sample.pdf",
            "version": 2,
            "file_type": "pdf",
        }
    )

    assert label == (
        "sample.pdf · v2 · PDF"
    )

    print(
        "Document label formatting: PASS"
    )

    print()
    print(
        "Pre-deployment Streamlit product UI tests passed."
    )


if __name__ == "__main__":
    main()