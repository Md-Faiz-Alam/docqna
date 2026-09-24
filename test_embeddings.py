from docqna.embeddings import SentenceTransformerEmbedding
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


# Load PDF
document = PdfLoader().load(
    "data/samples/sample.pdf"
)


# Preprocess
document = TextNormalizer().normalize_document(document)
document = TextCleaner().clean_document(document)
document = StructureDetector().detect(document)
document = MetadataBuilder().build(document)


# Chunk
chunker = FixedChunker(
    chunk_size=200,
    overlap=40,
)

chunks = chunker.chunk(document)


# Embedding model
embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# Embed all chunks
texts = [
    chunk.text
    for chunk in chunks
]

embeddings = embedding_model.embed_documents(texts)


# Validate
print("Document:", document.filename)
print("Chunks:", len(chunks))
print("Embeddings:", len(embeddings))
print("Embedding dimension:", embedding_model.dimension)


assert len(chunks) == len(embeddings)

assert all(
    len(vector) == embedding_model.dimension
    for vector in embeddings
)

print("Chunk/embedding count test passed")
print("Embedding dimension test passed")


print("\nFirst chunk:")
print(chunks[0].text[:300])

print("\nFirst 10 embedding values:")
print(embeddings[0][:10])