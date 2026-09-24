from pathlib import Path

from docqna.embeddings import SentenceTransformerEmbedding
from docqna.indexing import ChromaVectorStore
from docqna.ingestion.chunking import StructureAwareChunker
from docqna.ingestion.loaders import PdfLoader
from docqna.ingestion.preprocessing import (
    MetadataBuilder,
    StructureDetector,
    TextCleaner,
    TextNormalizer,
)
from docqna.settings import get_settings


settings = get_settings()


# --------------------------------
# 1. Separate structure index
# --------------------------------

STRUCTURE_INDEX_DIR = Path(
    "data/indexes/chroma_structure"
)

STRUCTURE_COLLECTION = (
    "docqna_structure_documents"
)


# --------------------------------
# 2. Load
# --------------------------------

document = PdfLoader().load(
    "data/samples/sample.pdf"
)


# --------------------------------
# 3. Preprocess
# --------------------------------

document = TextNormalizer().normalize_document(
    document
)

document = TextCleaner().clean_document(
    document
)

document = StructureDetector().detect(
    document
)

document = MetadataBuilder().build(
    document
)


# --------------------------------
# 4. Structure-aware chunking
# --------------------------------

chunker = StructureAwareChunker(
    chunk_size=200,
    overlap=40,
)


chunks = chunker.chunk(
    document
)


print(
    "Document:",
    document.filename,
)

print(
    "Structure-aware chunks generated:",
    len(chunks),
)


assert chunks


# --------------------------------
# 5. Validate chunk metadata
# --------------------------------

for chunk in chunks:
    assert (
        chunk.metadata.get(
            "chunking_strategy"
        )
        == "structure_aware"
    )

    assert (
        chunk.metadata.get(
            "word_count",
            0,
        )
        <= 200
    )


print(
    "Structure-aware chunk metadata: OK"
)


# --------------------------------
# 6. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 7. Embed structure chunks
# --------------------------------

embeddings = (
    embedding_model.embed_documents(
        [
            chunk.text
            for chunk in chunks
        ]
    )
)


print(
    "Embeddings generated:",
    len(embeddings),
)


assert (
    len(embeddings)
    == len(chunks)
)


# --------------------------------
# 8. Separate Chroma store
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=STRUCTURE_INDEX_DIR,
    collection_name=STRUCTURE_COLLECTION,
)


# --------------------------------
# 9. Index
# --------------------------------

vector_store.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
)


print(
    "Records in structure Chroma:",
    vector_store.count(),
)


assert (
    vector_store.count()
    == len(chunks)
)


# --------------------------------
# 10. Reconstruct stored chunks
# --------------------------------

stored_chunks = (
    vector_store.get_all_chunks()
)


assert (
    len(stored_chunks)
    == len(chunks)
)


for chunk in stored_chunks:
    assert (
        chunk.metadata.get(
            "chunking_strategy"
        )
        == "structure_aware"
    )


print(
    "Stored chunk reconstruction: OK"
)


# --------------------------------
# 11. Simple semantic sanity check
# --------------------------------

query = (
    "What techniques are used "
    "for identifying project risks?"
)


query_embedding = (
    embedding_model.embed_query(
        query
    )
)


results = (
    vector_store.query_by_embedding(
        embedding=query_embedding,
        top_k=5,
    )
)


documents = (
    results.get(
        "documents"
    )
)

metadatas = (
    results.get(
        "metadatas"
    )
)

distances = (
    results.get(
        "distances"
    )
)


if (
    documents is None
    or metadatas is None
    or distances is None
):
    raise RuntimeError(
        "Structure index query did not return "
        "documents, metadatas, and distances."
    )


print(
    "\nSemantic sanity query:"
)

print(
    query
)


for index, chunk_id in enumerate(
    results["ids"][0],
    start=1,
):
    metadata = (
        metadatas[0][index - 1]
    )

    text = (
        documents[0][index - 1]
        or ""
    )

    distance = (
        distances[0][index - 1]
    )

    print(
        "\n------------------------"
    )

    print(
        "Rank:",
        index,
    )

    print(
        "ID:",
        chunk_id,
    )

    print(
        "Distance:",
        distance,
    )

    print(
        "Pages:",
        metadata.get(
            "page_start"
        ),
        "-",
        metadata.get(
            "page_end"
        ),
    )

    print(
        "Strategy:",
        metadata.get(
            "chunking_strategy"
        ),
    )

    print(
        "Text:",
        text[:400],
    )


print(
    "\nStructure-aware index build passed"
)