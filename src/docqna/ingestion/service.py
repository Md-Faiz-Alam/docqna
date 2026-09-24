from dataclasses import dataclass
from pathlib import Path

from docqna.exceptions import UnsupportedFileTypeError
from docqna.ingestion.chunking import (
    BaseChunker,
    FixedChunker,
)
from docqna.ingestion.loaders import (
    BaseDocumentLoader,
    DocxLoader,
    PdfLoader,
    TxtLoader,
)
from docqna.ingestion.models import (
    DocumentChunk,
    LoadedDocument,
)
from docqna.ingestion.preprocessing import (
    MetadataBuilder,
    StructureDetector,
    TextCleaner,
    TextNormalizer,
)


@dataclass(frozen=True, slots=True)
class DocumentProcessingResult:
    """
    Result produced by the document processing pipeline.

    The document contains the fully preprocessed source
    representation. Chunks contain the retrieval units
    ready for embedding and indexing.
    """

    document: LoadedDocument
    chunks: list[DocumentChunk]


class DocumentIngestionService:
    """
    Reusable production document-processing service.

    Pipeline:

    load
        -> normalize
        -> clean
        -> structure detection
        -> metadata enrichment
        -> chunking

    The production chunking configuration remains
    FixedChunker(chunk_size=200, overlap=40).
    """

    def __init__(
        self,
        loaders: tuple[
            BaseDocumentLoader,
            ...,
        ]
        | None = None,
        chunker: BaseChunker | None = None,
        normalizer: TextNormalizer | None = None,
        cleaner: TextCleaner | None = None,
        structure_detector: StructureDetector | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self.loaders = loaders or (
            PdfLoader(),
            DocxLoader(),
            TxtLoader(),
        )

        self.chunker = chunker or FixedChunker(
            chunk_size=200,
            overlap=40,
        )

        self.normalizer = (
            normalizer
            or TextNormalizer()
        )

        self.cleaner = (
            cleaner
            or TextCleaner()
        )

        self.structure_detector = (
            structure_detector
            or StructureDetector()
        )

        self.metadata_builder = (
            metadata_builder
            or MetadataBuilder()
        )

    def _get_loader(
        self,
        path: str | Path,
    ) -> BaseDocumentLoader:
        """
        Return the loader capable of processing the file.
        """

        file_path = Path(path)

        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader

        extension = (
            file_path.suffix.lower()
            or "<none>"
        )

        raise UnsupportedFileTypeError(
            "No document loader is registered for "
            f"file type '{extension}'."
        )

    def load(
        self,
        path: str | Path,
    ) -> LoadedDocument:
        """
        Load a supported source document.
        """

        loader = self._get_loader(path)

        return loader.load(path)

    def preprocess(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """
        Apply the production preprocessing pipeline.
        """

        document = (
            self.normalizer.normalize_document(
                document
            )
        )

        document = (
            self.cleaner.clean_document(
                document
            )
        )

        document = (
            self.structure_detector.detect(
                document
            )
        )

        document = (
            self.metadata_builder.build(
                document
            )
        )

        return document

    def process(
        self,
        path: str | Path,
    ) -> DocumentProcessingResult:
        """
        Load, preprocess, and chunk one document.
        """

        document = self.load(path)

        document = self.preprocess(
            document
        )

        chunks = self.chunker.chunk(
            document
        )

        return DocumentProcessingResult(
            document=document,
            chunks=chunks,
        )