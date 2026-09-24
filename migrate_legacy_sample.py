from copy import deepcopy
from pathlib import Path
from typing import cast

from chromadb.api.types import Metadata

from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.indexing import ChromaVectorStore
from docqna.ingestion import DocumentRegistry


EXPECTED_CHUNK_COUNT = 192

SAMPLE_PATH = Path(
    "data/samples/sample.pdf"
)


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Sample document not found: "
            f"{SAMPLE_PATH}"
        )

    registry = DocumentRegistry()

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_INDEX_DIR,
        collection_name=DEFAULT_CHROMA_COLLECTION,
    )

    # ---------------------------------
    # Verify frozen production corpus
    # ---------------------------------

    current_count = vector_store.count()

    if current_count != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            "Legacy migration expected exactly "
            f"{EXPECTED_CHUNK_COUNT} chunks, "
            f"but found {current_count}. "
            "Do not continue automatically."
        )

    content_hash = (
        registry.compute_content_hash(
            SAMPLE_PATH
        )
    )

    existing_records = (
        registry.list_records()
    )

    existing_record = next(
        (
            record
            for record in existing_records
            if record.content_hash
            == content_hash
        ),
        None,
    )

    # ---------------------------------
    # Idempotency check
    # ---------------------------------

    if existing_record is not None:
        print(
            "Sample document is already registered."
        )
        print(
            "Document ID:",
            existing_record.document_id,
        )
        print(
            "Version:",
            existing_record.version,
        )
        return

    # ---------------------------------
    # Prepare document identity
    # ---------------------------------

    record = registry.prepare(
        SAMPLE_PATH
    )

    # ---------------------------------
    # Read existing Chroma metadata
    # ---------------------------------

    results = vector_store.collection.get(
        include=[
            "metadatas",
        ]
    )

    ids = results.get(
        "ids"
    ) or []

    metadatas = results.get(
        "metadatas"
    ) or []

    if (
        len(ids) != EXPECTED_CHUNK_COUNT
        or len(metadatas)
        != EXPECTED_CHUNK_COUNT
    ):
        raise RuntimeError(
            "Chroma returned an unexpected number "
            "of legacy chunks."
        )

    original_metadatas = deepcopy(
        metadatas
    )

    updated_metadatas: list[Metadata] = []

    # ---------------------------------
    # Validate and enrich each chunk
    # ---------------------------------

    for chunk_id, raw_metadata in zip(
        ids,
        metadatas,
    ):
        metadata = dict(
            raw_metadata or {}
        )

        filename = metadata.get(
            "filename"
        )

        if filename != SAMPLE_PATH.name:
            raise RuntimeError(
                "Legacy production index contains "
                f"unexpected filename: {filename!r}"
            )

        strategy = metadata.get(
            "chunking_strategy"
        )

        if strategy != "fixed":
            raise RuntimeError(
                "Legacy production index contains "
                "a non-fixed chunking strategy: "
                f"{strategy!r}"
            )

        metadata[
            "document_id"
        ] = record.document_id

        metadata[
            "document_version"
        ] = record.version

        metadata[
            "content_hash"
        ] = record.content_hash

        metadata[
            "original_chunk_id"
        ] = str(
            chunk_id
        )

        updated_metadatas.append(
            cast(
                Metadata,
                metadata,
            )
        )

    # ---------------------------------
    # Update metadata only
    # ---------------------------------

    try:
        vector_store.collection.update(
            ids=[
                str(chunk_id)
                for chunk_id in ids
            ],
            metadatas=updated_metadatas,
        )

        # Verify corpus size did not change.
        if (
            vector_store.count()
            != EXPECTED_CHUNK_COUNT
        ):
            raise RuntimeError(
                "Chunk count changed during "
                "legacy metadata migration."
            )

        # Verify reconstruction through our
        # production vector-store abstraction.
        migrated_chunks = (
            vector_store.get_all_chunks()
        )

        if (
            len(migrated_chunks)
            != EXPECTED_CHUNK_COUNT
        ):
            raise RuntimeError(
                "Could not reconstruct all migrated "
                "production chunks."
            )

        if not all(
            chunk.metadata.get(
                "document_id"
            )
            == record.document_id
            for chunk in migrated_chunks
        ):
            raise RuntimeError(
                "document_id migration verification "
                "failed."
            )

        if not all(
            chunk.metadata.get(
                "document_version"
            )
            == 1
            for chunk in migrated_chunks
        ):
            raise RuntimeError(
                "document_version migration "
                "verification failed."
            )

        if not all(
            chunk.metadata.get(
                "content_hash"
            )
            == record.content_hash
            for chunk in migrated_chunks
        ):
            raise RuntimeError(
                "content_hash migration verification "
                "failed."
            )

        # Only register the document after the
        # Chroma metadata migration succeeds.
        registry.commit(
            record
        )

    except Exception:
        # Restore original metadata if anything
        # fails before migration completes.
        rollback_metadatas: list[Metadata] = [
            cast(
                Metadata,
                dict(metadata or {}),
            )
            for metadata
            in original_metadatas
        ]

        vector_store.collection.update(
            ids=[
                str(chunk_id)
                for chunk_id in ids
            ],
            metadatas=rollback_metadatas,
        )

        raise

    print(
        "Legacy production migration: PASS"
    )
    print(
        "Document ID:",
        record.document_id,
    )
    print(
        "Version:",
        record.version,
    )
    print(
        "Content hash:",
        record.content_hash,
    )
    print(
        "Chunks migrated:",
        vector_store.count(),
    )
    print(
        "Registry records:",
        len(
            registry.list_records()
        ),
    )


if __name__ == "__main__":
    main()