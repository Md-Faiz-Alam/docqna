"""Session-scoped hybrid retrieval with per-session BM25 indexes.

Each anonymous session gets its own BM25 index built only from that
session's chunks. Uploads and deletes invalidate just the affected
session, so one user's activity never forces a rebuild over the whole
corpus. Dense retrieval stays shared (Chroma filters by session_id).
"""

import logging
from collections import OrderedDict
from collections.abc import Callable
from threading import Lock

from docqna.filters import MetadataFilters, normalize_metadata_filters
from docqna.ingestion.models import DocumentChunk
from docqna.observability import BM25_INDEX_BUILDS
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.bm25 import BM25Retriever
from docqna.retrieval.hybrid import RRFHybridRetriever
from docqna.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)

ChunkLoader = Callable[[str | None], list[DocumentChunk]]

# Key used for unscoped (internal / evaluation) retrieval.
_GLOBAL_SCOPE = "__all__"


class SessionScopedHybridRetriever(BaseRetriever):
    """Dense + per-session BM25, fused with RRF.

    `load_chunks(session_id)` returns the chunks for one session
    (or every chunk when session_id is None). BM25 indexes are cached in a
    bounded LRU and rebuilt lazily after `invalidate()`.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        load_chunks: ChunkLoader,
        *,
        rrf_k: int = 60,
        candidate_pool_size: int = 10,
        max_cached_sessions: int = 256,
    ) -> None:
        if max_cached_sessions <= 0:
            raise ValueError("max_cached_sessions must be greater than 0.")

        self.dense_retriever = dense_retriever
        self.load_chunks = load_chunks
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size
        self.max_cached_sessions = max_cached_sessions

        self._cache: OrderedDict[str, BM25Retriever | None] = OrderedDict()
        self._generations: dict[str, int] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, session_id: str | None = None) -> None:
        """Drop one session's BM25 index, or every index when session_id is None."""

        with self._lock:
            if session_id is None:
                keys = [*self._cache, _GLOBAL_SCOPE]
                self._cache.clear()
            else:
                keys = [session_id, _GLOBAL_SCOPE]
                self._cache.pop(session_id, None)
                self._cache.pop(_GLOBAL_SCOPE, None)

            for key in keys:
                self._generations[key] = self._generations.get(key, 0) + 1

    def cached_sessions(self) -> int:
        with self._lock:
            return len(self._cache)

    def _bm25_for(self, session_id: str | None) -> BM25Retriever | None:
        key = session_id or _GLOBAL_SCOPE

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            generation = self._generations.get(key, 0)

        # Build outside the lock so slow loads never block other sessions.
        chunks = self.load_chunks(session_id)
        index = BM25Retriever(chunks=chunks) if chunks else None
        BM25_INDEX_BUILDS.inc()
        logger.info(
            "bm25_session_index_built scope=%s chunks=%d",
            "session" if session_id else "global",
            len(chunks),
        )

        with self._lock:
            # Skip caching if the session changed while we were building.
            if self._generations.get(key, 0) == generation:
                self._cache[key] = index
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_cached_sessions:
                    self._cache.popitem(last=False)

        return index

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        metadata_filters = normalize_metadata_filters(metadata_filters)
        session_id = metadata_filters.get("session_id") if metadata_filters else None
        if session_id is not None and not isinstance(session_id, str):
            raise ValueError("session_id filter must be a string.")

        bm25 = self._bm25_for(session_id)
        if bm25 is None:
            # Nothing indexed for this scope: let the pipeline abstain.
            return []

        hybrid = RRFHybridRetriever(
            dense_retriever=self.dense_retriever,
            sparse_retriever=bm25,
            rrf_k=self.rrf_k,
            candidate_pool_size=self.candidate_pool_size,
        )
        return hybrid.retrieve(query=query, top_k=top_k, metadata_filters=metadata_filters)
