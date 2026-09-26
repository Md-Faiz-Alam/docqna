class DocQNAError(Exception):
    """Base exception for all DocQ&A errors."""


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

class ConfigurationError(DocQNAError):
    """Raised when application configuration is invalid."""


# ---------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------

class DocumentError(DocQNAError):
    """Base exception for document-processing errors."""


class DocumentValidationError(DocumentError):
    """Raised when a document fails validation."""


class UnsupportedFileTypeError(DocumentValidationError):
    """Raised when an unsupported document type is provided."""


class DuplicateDocumentError(DocumentValidationError):
    """Raised when an identical document has already been ingested."""


class ResourceLimitError(DocumentError):
    """Raised when a public deployment resource limit is exceeded."""


class DocumentLoadError(DocumentError):
    """Raised when document contents cannot be loaded."""

class ChunkingError(DocumentError):
    """Raised when a document cannot be chunked."""

class EmbeddingError(DocQNAError):
    """Raised when embedding generation fails."""

class IndexingError(DocQNAError):
    """Raised when vector indexing or storage fails."""


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

class RetrievalError(DocQNAError):
    """Raised when retrieval fails."""


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

class GenerationError(DocQNAError):
    """Raised when answer generation fails."""

class LLMModelError(GenerationError):
    """Raised when the configured LLM model is unavailable."""

class LLMConnectionError(GenerationError):
    """Raised when the configured LLM provider cannot be reached."""


# ---------------------------------------------------------
# Citations
# ---------------------------------------------------------

class CitationError(DocQNAError):
    """Raised when citation processing or validation fails."""
