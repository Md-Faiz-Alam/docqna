import logging

from docqna.api.dependencies import invalidate_session_index
from docqna.indexing import DocumentIndexService

logger = logging.getLogger(__name__)


def purge_expired_documents(service: DocumentIndexService) -> int:
    """Apply the anonymous-document retention policy before serving a request."""

    deleted_count = service.purge_expired_documents()
    if deleted_count > 0:
        invalidate_session_index()
        logger.info("document_retention_cache_invalidated deleted=%d", deleted_count)
    return deleted_count
