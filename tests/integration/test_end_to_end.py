"""End-to-end RAG checks with real models on a self-contained fixture.

Run with: pytest -m integration
The first run downloads the embedding and cross-encoder models. The LLM
test is skipped automatically when Ollama is not reachable.
"""

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from docqna.api.dependencies import build_llm, build_rag_pipeline
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.generation.providers.base import BaseLLMProvider
from docqna.indexing import ChromaVectorStore, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRegistry
from docqna.rag import RAGPipeline
from docqna.rag.pipeline import ABSTENTION_ANSWER
from docqna.settings import Settings

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "employee_handbook.txt"
SESSION_ID = "integration-session-" + "b" * 32


class EchoFirstSourceLLM(BaseLLMProvider):
    """Deterministic LLM that answers from source [1] with a citation."""

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return "The answer is supported by the first source [1]."

    def health_check(self) -> bool:
        return True


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture(scope="module")
def indexed_store(
    settings: Settings, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[tuple[SentenceTransformerEmbedding, ChromaVectorStore]]:
    root = tmp_path_factory.mktemp("e2e")
    embedding_model = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = ChromaVectorStore(persist_directory=root / "chroma", collection_name="e2e")
    service = DocumentIndexService(
        ingestion_service=DocumentIngestionService(),
        registry=DocumentRegistry(root / "documents.json"),
        embedding_model=embedding_model,
        vector_store=vector_store,
    )
    document = root / FIXTURE.name
    shutil.copy(FIXTURE, document)

    result = service.index_document(document, session_id=SESSION_ID)
    assert result.chunk_count > 0

    yield embedding_model, vector_store


def make_pipeline(
    settings: Settings,
    indexed_store: tuple[SentenceTransformerEmbedding, ChromaVectorStore],
    llm: BaseLLMProvider,
) -> RAGPipeline:
    embedding_model, vector_store = indexed_store
    return build_rag_pipeline(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        cross_encoder_model=None,
        llm=llm,
    )


def test_answerable_question_is_grounded_and_cited(settings, indexed_store) -> None:
    pipeline = make_pipeline(settings, indexed_store, EchoFirstSourceLLM())

    response = pipeline.ask_with_citations(
        "How many days of annual leave can be carried over?",
        metadata_filters={"session_id": SESSION_ID},
    )

    assert response.answer != ABSTENTION_ANSWER
    assert response.cited_source_ids == [1]
    assert "carried over" in response.sources[0].text


def test_unrelated_question_abstains(settings, indexed_store) -> None:
    pipeline = make_pipeline(settings, indexed_store, EchoFirstSourceLLM())

    response = pipeline.ask_with_citations(
        "What is the boiling point of mercury in kelvin?",
        metadata_filters={"session_id": SESSION_ID},
    )

    assert response.answer == ABSTENTION_ANSWER
    assert response.sources == []


def test_other_session_sees_nothing(settings, indexed_store) -> None:
    pipeline = make_pipeline(settings, indexed_store, EchoFirstSourceLLM())

    response = pipeline.ask_with_citations(
        "How many days of annual leave can be carried over?",
        metadata_filters={"session_id": "another-session-" + "c" * 32},
    )

    assert response.answer == ABSTENTION_ANSWER


def test_real_llm_answer_cites_valid_sources(settings, indexed_store) -> None:
    llm = build_llm(settings)
    if not llm.health_check():
        pytest.skip("Ollama is not reachable")

    pipeline = make_pipeline(settings, indexed_store, llm)
    response = pipeline.ask_with_citations(
        "What is the daily meal expense cap while travelling?",
        metadata_filters={"session_id": SESSION_ID},
    )

    if response.answer == ABSTENTION_ANSWER:
        pytest.fail("Model abstained on an answerable question")
    assert response.cited_source_ids
    assert all(1 <= cid <= len(response.sources) for cid in response.cited_source_ids)
    assert re.search(r"\b60\b", response.answer)
