from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.indexing import ChromaVectorStore
from docqna.ingestion.chunking import FixedChunker
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
# 1. Load
# --------------------------------

document = PdfLoader().load(
    "data/samples/sample.pdf"
)


# --------------------------------
# 2. Preprocess
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
# 3. Chunk
# --------------------------------

chunker = FixedChunker(
    chunk_size=200,
    overlap=40,
)

chunks = chunker.chunk(
    document
)


# --------------------------------
# 4. Embed
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)

embeddings = embedding_model.embed_documents(
    [
        chunk.text
        for chunk in chunks
    ]
)


# --------------------------------
# 5. Chroma store
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=CHROMA_INDEX_DIR,
    collection_name=DEFAULT_CHROMA_COLLECTION,
)


# --------------------------------
# 6. Index
# --------------------------------

vector_store.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
)


print("Document:", document.filename)
print("Chunks generated:", len(chunks))
print(
    "Records in Chroma:",
    vector_store.count(),
)


assert vector_store.count() == len(chunks)

print("Index count test passed")


# --------------------------------
# 7. Semantic query
# --------------------------------

query = (
    "What techniques are used "
    "for identifying project risks?"
)

query_embedding = embedding_model.embed_query(
    query
)

results = vector_store.query_by_embedding(
    embedding=query_embedding,
    top_k=5,
)


# --------------------------------
# 8. Validate returned fields
# --------------------------------

ids = results["ids"]
distances = results.get("distances")
metadatas = results.get("metadatas")
documents = results.get("documents")


if (
    distances is None
    or metadatas is None
    or documents is None
):
    raise RuntimeError(
        "Chroma query did not return "
        "documents, metadatas, and distances."
    )


# --------------------------------
# 9. Display results
# --------------------------------

print("\nQuery:")
print(query)


for index, chunk_id in enumerate(ids[0]):
    document_text = documents[0][index]

    if document_text is None:
        document_text = ""

    print("\n------------------------")

    print(
        "ID:",
        chunk_id,
    )

    print(
        "Distance:",
        distances[0][index],
    )

    print(
        "Metadata:",
        metadatas[0][index],
    )

    print(
        "Text:",
        document_text[:500],
    )