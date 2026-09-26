from sentence_transformers import CrossEncoder

from docqna.exceptions import RetrievalError
from docqna.filters import (
    MetadataFilters,
    normalize_metadata_filters,
)
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class CrossEncoderReranker(BaseRetriever):
    """
    Retrieve a larger candidate set and rerank it
    with a sentence-transformers CrossEncoder.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str = (
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        candidate_pool_size: int = 20,
        device: str = "cpu",
        model: CrossEncoder | None = None,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError(
                "candidate_pool_size must be "
                "greater than 0."
            )

        self.base_retriever = (
            base_retriever
        )

        self.model_name = (
            model_name
        )

        self.candidate_pool_size = (
            candidate_pool_size
        )

        if model is not None:
            self.model = model

        else:
            try:
                self.model = CrossEncoder(
                    model_name,
                    device=device,
                )

            except Exception as exc:
                raise RetrievalError(
                    "Could not load cross-encoder "
                    f"model '{model_name}'."
                ) from exc

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve candidates and rerank them."""

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
            candidates = (
                self.base_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                    metadata_filters=metadata_filters,
                )
            )

        else:
            candidates = (
                self.base_retriever.retrieve(
                    query=query,
                    top_k=candidate_k,
                )
            )

        if not candidates:
            return []

        try:
            pairs = [
                (
                    query,
                    candidate.text,
                )
                for candidate in candidates
            ]

            raw_scores = self.model.predict(
                pairs
            )

        except Exception as exc:
            raise RetrievalError(
                "Cross-encoder reranking failed."
            ) from exc

        scored_candidates: list[
            tuple[
                RetrievalResult,
                float,
            ]
        ] = []

        for candidate, raw_score in zip(
            candidates,
            raw_scores,
        ):
            score = float(
                raw_score
            )

            scored_candidates.append(
                (
                    candidate,
                    score,
                )
            )

        scored_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].rank,
                item[0].chunk_index,
                item[0].chunk_id,
            )
        )

        results: list[
            RetrievalResult
        ] = []

        for rank, (
            source,
            score,
        ) in enumerate(
            scored_candidates[:top_k],
            start=1,
        ):
            results.append(
                RetrievalResult(
                    chunk_id=source.chunk_id,
                    text=source.text,
                    rank=rank,

                    # Preserve our shared convention:
                    # lower distance = better.
                    distance=-score,

                    score=score,

                    retrieval_method=(
                        "cross_encoder_rerank"
                    ),

                    filename=source.filename,
                    file_type=source.file_type,
                    chunk_index=(
                        source.chunk_index
                    ),

                    document_id=(
                        source.document_id
                    ),

                    document_version=(
                        source.document_version
                    ),

                    page_start=(
                        source.page_start
                    ),

                    page_end=(
                        source.page_end
                    ),

                    section=(
                        source.section
                    ),

                    source_segment_indices=(
                        source.source_segment_indices
                    ),

                    sections=(
                        source.sections
                    ),
                )
            )

        return results