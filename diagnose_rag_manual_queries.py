from docqna.api.dependencies import (
    get_document_index_service,
    get_rag_pipeline,
)


QUERIES = [
    (
        "scheduling",
        (
            "Which tools and techniques are described "
            "for project scheduling and estimating "
            "task durations?"
        ),
    ),
    (
        "ethics",
        (
            "What are the four pillars of the PMI Code "
            "of Ethics and Professional Conduct?"
        ),
    ),
]


def page_label(
    page_start: int | None,
    page_end: int | None,
) -> str:
    if page_start is None:
        return "unknown"

    if (
        page_end is None
        or page_end == page_start
    ):
        return str(
            page_start
        )

    return (
        f"{page_start}-{page_end}"
    )


def main() -> None:
    service = (
        get_document_index_service()
    )

    documents = (
        service.list_documents()
    )

    sample = next(
        (
            document
            for document in documents
            if document.filename.casefold()
            == "sample.pdf"
        ),
        None,
    )

    if sample is None:
        raise RuntimeError(
            "sample.pdf is not currently "
            "registered/indexed."
        )

    print(
        "Diagnostic document:"
    )
    print(
        f"  filename={sample.filename}"
    )
    print(
        f"  document_id={sample.document_id}"
    )
    print(
        f"  version={sample.version}"
    )

    pipeline = get_rag_pipeline()

    for name, query in QUERIES:
        print()
        print(
            "=" * 72
        )
        print(
            f"{name.upper()} RETRIEVAL"
        )
        print(
            "=" * 72
        )
        print(
            query
        )

        results = (
            pipeline.retriever.retrieve(
                query=query,
                top_k=5,
                metadata_filters={
                    "document_id": (
                        sample.document_id
                    ),
                },
            )
        )

        for result in results:
            snippet = (
                " ".join(
                    result.text.split()
                )[:500]
            )

            print()
            print(
                (
                    f"rank={result.rank} "
                    f"score={result.score} "
                    f"pages="
                    f"{page_label(result.page_start, result.page_end)}"
                )
            )
            print(
                f"chunk_id={result.chunk_id}"
            )
            print(
                snippet
            )

    print()
    print(
        "Retrieval diagnostic completed."
    )


if __name__ == "__main__":
    main()
