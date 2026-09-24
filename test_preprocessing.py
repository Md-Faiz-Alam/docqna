from docqna.ingestion.loaders import TxtLoader
from docqna.ingestion.loaders import DocxLoader
from docqna.ingestion.preprocessing import (
    MetadataBuilder,
    StructureDetector,
    TextCleaner,
    TextNormalizer,
)


from docqna.ingestion.loaders import PdfLoader

document = PdfLoader().load(
    "data/samples/sample.pdf"
)

normalizer = TextNormalizer()
cleaner = TextCleaner()
structure_detector = StructureDetector()
metadata_builder = MetadataBuilder()


document = normalizer.normalize_document(document)

document = cleaner.clean_document(document)

document = structure_detector.detect(document)

document = metadata_builder.build(document)


print("Filename:", document.filename)
print("Type:", document.file_type)
print("Segments:", document.segment_count)

print("\nDocument metadata:")
print(document.metadata)


for segment in document.segments:
    print("\n--------------------")
    print("Index:", segment.segment_index)
    print("Page number:", segment.page_number)
    print("Section:", segment.section)
    print("Text:", segment.text)
    print("Metadata:", segment.metadata)