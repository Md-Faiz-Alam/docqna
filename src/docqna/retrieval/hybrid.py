from docqna.filters import (
    MetadataFilters,
    normalize_metadata_filters,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class RRFHybridRetriever(BaseRetriever):
    """
    Hybrid dense + BM25 retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        rrf_k: int = 60,
        candidate_pool_size: int = 10,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

        if candidate_pool_size <= 0:
            raise ValueError(
                "candidate_pool_size must be greater than 0."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
        self.candidate_pool_size = candidate_pool_size

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and fuse dense + sparse rankings."""

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        metadata_filters = normalize_metadata_filters(
            metadata_filters
        )

        candidate_k = max(
            top_k,
            self.candidate_pool_size,
        )

        if metadata_filters:
            dense_results = (
                self.dense_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                    metadata_filters=metadata_filters,
                )
            )

            sparse_results = (
                self.sparse_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                    metadata_filters=metadata_filters,
                )
            )

        else:
            dense_results = (
                self.dense_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                )
            )

            sparse_results = (
                self.sparse_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                )
            )

        fusion_scores: dict[
            str,
            float,
        ] = {}

        result_lookup: dict[
            str,
            RetrievalResult,
        ] = {}

        best_source_rank: dict[
            str,
            int,
        ] = {}

        for result_list in (
            dense_results,
            sparse_results,
        ):
            for result in result_list:
                chunk_id = result.chunk_id

                fusion_scores[
                    chunk_id
                ] = (
                    fusion_scores.get(
                        chunk_id,
                        0.0,
                    )
                    + 1.0
                    / (
                        self.rrf_k
                        + result.rank
                    )
                )

                if chunk_id not in result_lookup:
                    result_lookup[
                        chunk_id
                    ] = result

                current_best = (
                    best_source_rank.get(
                        chunk_id
                    )
                )

                if (
                    current_best is None
                    or result.rank < current_best
                ):
                    best_source_rank[
                        chunk_id
                    ] = result.rank

        ranked_chunk_ids = sorted(
            fusion_scores,
            key=lambda chunk_id: (
                -fusion_scores[
                    chunk_id
                ],
                best_source_rank[
                    chunk_id
                ],
                result_lookup[
                    chunk_id
                ].chunk_index,
                chunk_id,
            ),
        )

        results: list[
            RetrievalResult
        ] = []

        for rank, chunk_id in enumerate(
            ranked_chunk_ids[:top_k],
            start=1,
        ):
            source = result_lookup[
                chunk_id
            ]

            rrf_score = fusion_scores[
                chunk_id
            ]

            results.append(
                RetrievalResult(
                    chunk_id=source.chunk_id,
                    text=source.text,
                    rank=rank,

                    # Preserve the shared convention
                    # where smaller distance is better.
                    distance=-rrf_score,

                    score=rrf_score,
                    retrieval_method="hybrid_rrf",

                    filename=source.filename,
                    file_type=source.file_type,
                    chunk_index=source.chunk_index,

                    document_id=source.document_id,
                    document_version=(
                        source.document_version
                    ),

                    page_start=source.page_start,
                    page_end=source.page_end,
                    section=source.section,

                    source_segment_indices=(
                        source.source_segment_indices
                    ),

                    sections=source.sections,
                )
            )

        return results