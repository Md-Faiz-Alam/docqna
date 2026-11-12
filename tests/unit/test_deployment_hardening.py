import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from sentence_transformers import CrossEncoder

import docqna.api.dependencies as dependencies
from docqna.constants import APP_VERSION
from docqna.filters import MetadataFilters
from docqna.indexing import ChromaVectorStore
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult
from docqna.retrieval.reranker import CrossEncoderReranker


class StubRetriever(BaseRetriever):
    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: MetadataFilters | None = None
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk_1",
                text="Relevant deployment hardening evidence.",
                rank=1,
                distance=0.0,
                filename="sample.txt",
                file_type="txt",
                chunk_index=0,
            )
        ][:top_k]


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert pairs
        return [1.25 for _ in pairs]


def test_injected_reranker_model() -> None:
    fake_model = FakeCrossEncoder()

    with patch(
        "docqna.retrieval.reranker.CrossEncoder",
        side_effect=AssertionError("CrossEncoder should not reload when a model is injected."),
    ):
        reranker = CrossEncoderReranker(
            base_retriever=StubRetriever(),
            candidate_pool_size=1,
            model=cast(CrossEncoder, fake_model),
        )

    results = reranker.retrieve(query="What evidence is relevant?", top_k=1)

    assert reranker.model is fake_model
    assert len(results) == 1
    assert results[0].score == 1.25


def test_dependency_model_cache_survives_pipeline_clear() -> None:
    dependencies.get_cross_encoder_model.cache_clear()
    fake_model = FakeCrossEncoder()

    try:
        with patch.object(
            dependencies, "CrossEncoder", return_value=cast(CrossEncoder, fake_model)
        ) as factory:
            first = dependencies.get_cross_encoder_model()
            second = dependencies.get_cross_encoder_model()

            dependencies.clear_rag_pipeline_cache()

            third = dependencies.get_cross_encoder_model()

            assert first is fake_model
            assert second is first
            assert third is first
            assert factory.call_count == 1

    finally:
        dependencies.get_cross_encoder_model.cache_clear()


def test_chroma_telemetry_disabled() -> None:
    with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        store = ChromaVectorStore(
            persist_directory=Path(temp_dir) / "chroma", collection_name="deployment_hardening_test"
        )

        settings = store.client.get_settings()
        assert settings.anonymized_telemetry is False


def test_package_version_matches_app_version() -> None:
    with Path("pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)

    assert project["project"]["version"] == APP_VERSION
