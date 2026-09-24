from docqna.ingestion.loaders import PdfLoader


loader = PdfLoader()

document = loader.load("data/samples/olist ecommerce.pdf")

print("Filename:", document.filename)
print("File type:", document.file_type)
print("Segments:", document.segment_count)
print("Total pages:", document.metadata["total_pages"])

for segment in document.segments[:3]:
    print("\n--- Page ---")
    print("Page number:", segment.page_number)
    print("Text:", segment.text[:500])