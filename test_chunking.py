from docqna.ingestion.chunking import FixedChunker
from docqna.ingestion.loaders import PdfLoader
from docqna.ingestion.preprocessing import (
    MetadataBuilder,
    StructureDetector,
    TextCleaner,
    TextNormalizer,
)


document = PdfLoader().load(
    "data/samples/sample.pdf"
)

document = TextNormalizer().normalize_document(document)
document = TextCleaner().clean_document(document)
document = StructureDetector().detect(document)
document = MetadataBuilder().build(document)


chunker = FixedChunker(
    chunk_size=500,
    overlap=50,
)

chunks = chunker.chunk(document)


if len(chunks) >= 2:
    overlap = 50

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-overlap:] == second_words[:overlap]

    print("Overlap test passed")


print("Document:", document.filename)
print("Chunks:", len(chunks))


for chunk in chunks:
    print("\n------------------------")
    print("Chunk ID:", chunk.chunk_id)
    print("Index:", chunk.chunk_index)
    print("Page:", chunk.page_start, "-", chunk.page_end)
    print("Section:", chunk.section)
    print("Source segments:", chunk.source_segment_indices)
    print("Metadata:", chunk.metadata)
    print("Text:", chunk.text)