import json
from collections.abc import Iterator

from fastapi.testclient import TestClient

from docqna.api.dependencies import get_rag_pipeline_provider
from docqna.api.main import app
from docqna.exceptions import GenerationError
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline, RAGResponse
from docqna.rag.models import CitationSource
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


class OneResultRetriever(BaseRetriever):
    def retrieve(self, query, top_k=5, metadata_filters=None) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="c1",
                text="Meals are capped at 60 euros per day.",
                rank=1,
                distance=0.1,
                score=3.0,
                filename="handbook.txt",
                file_type="txt",
                chunk_index=0,
            )
        ]


class StreamingLLM(BaseLLMProvider):
    def __init__(self, parts: list[str], repair: str = "") -> None:
        self.parts = parts
        self.repair = repair

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.repair

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        yield from self.parts

    def health_check(self) -> bool:
        return True


def collect(pipeline: RAGPipeline) -> tuple[list[str], RAGResponse]:
    items = list(pipeline.stream_with_citations("What is the meal cap?"))
    *tokens, final = items
    assert isinstance(final, RAGResponse)
    return [t for t in tokens if isinstance(t, str)], final


def test_pipeline_streams_tokens_then_validated_answer() -> None:
    pipeline = RAGPipeline(OneResultRetriever(), StreamingLLM(["Meals are capped ", "at 60 [1]."]))

    tokens, final = collect(pipeline)

    assert tokens == ["Meals are capped ", "at 60 [1]."]
    assert final.answer == "Meals are capped at 60 [1]."
    assert final.cited_source_ids == [1]


def test_stream_final_answer_replaces_draft_after_citation_repair() -> None:
    llm = StreamingLLM(["Meals are capped at 60 [7]."], repair="Meals are capped at 60 [1].")
    pipeline = RAGPipeline(OneResultRetriever(), llm)

    _, final = collect(pipeline)

    assert final.answer == "Meals are capped at 60 [1]."


def test_stream_abstains_when_citations_cannot_be_repaired() -> None:
    llm = StreamingLLM(["Unsupported claim."], repair="Still no citations.")
    pipeline = RAGPipeline(OneResultRetriever(), llm)

    _, final = collect(pipeline)

    assert final.answer == ABSTENTION_ANSWER


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class FakeStreamingPipeline:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def stream_with_citations(self, query, top_k=5, metadata_filters=None, history=None):
        yield "Draft "
        if self.fail:
            raise GenerationError("Ollama generation failed.")
        yield RAGResponse(
            answer="Final [1].",
            cited_source_ids=[1],
            sources=[
                CitationSource(
                    citation_id=1,
                    chunk_id="c1",
                    rank=1,
                    filename="handbook.txt",
                    file_type="txt",
                    distance=0.1,
                    text="evidence",
                )
            ],
        )


def test_stream_endpoint_emits_tokens_then_final(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline_provider] = lambda: FakeStreamingPipeline

    response = client.post("/qa/stream", json={"question": "Meal cap?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[0] == ("token", {"text": "Draft "})
    assert events[-1][0] == "final"
    assert events[-1][1]["answer"] == "Final [1]."
    assert events[-1][1]["abstained"] is False


def test_stream_endpoint_reports_mid_stream_errors_as_event(client: TestClient) -> None:
    app.dependency_overrides[get_rag_pipeline_provider] = lambda: (
        lambda: FakeStreamingPipeline(fail=True)
    )

    response = client.post("/qa/stream", json={"question": "Meal cap?"})

    events = parse_sse(response.text)
    assert events[-1] == (
        "error",
        {"error": "generation_error", "detail": "Ollama generation failed."},
    )


def test_stream_endpoint_rejects_invalid_requests_before_streaming(client: TestClient) -> None:
    response = client.post("/qa/stream", json={"question": "x" * 5000})

    assert response.status_code == 422
