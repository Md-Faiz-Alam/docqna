from fastapi.testclient import TestClient

from docqna.api.dependencies import get_rag_pipeline_provider
from docqna.api.main import app
from docqna.exceptions import GenerationError
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline, RAGResponse
from docqna.rag.prompt import build_condense_question_prompt
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult

HISTORY = [
    ("user", "What is the meal allowance when travelling?"),
    ("assistant", "Meals are capped at 60 euros per day [1]."),
]


class RecordingRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query, top_k=5, metadata_filters=None) -> list[RetrievalResult]:
        self.queries.append(query)
        return [
            RetrievalResult(
                chunk_id="c1",
                text="Economy class is the default for flights shorter than six hours.",
                rank=1,
                distance=0.1,
                filename="handbook.txt",
                file_type="txt",
                chunk_index=0,
            )
        ]


class ScriptedLLM(BaseLLMProvider):
    """Returns `rewrite` for the condense prompt and `answer` otherwise."""

    def __init__(self, rewrite: str | Exception, answer: str = "Economy class [1].") -> None:
        self.rewrite = rewrite
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self.prompts.append(prompt)
        if "FOLLOW-UP QUESTION" in prompt:
            if isinstance(self.rewrite, Exception):
                raise self.rewrite
            return self.rewrite
        return self.answer

    def health_check(self) -> bool:
        return True


def test_follow_up_is_rewritten_before_retrieval() -> None:
    retriever = RecordingRetriever()
    llm = ScriptedLLM("STANDALONE QUESTION: Which flight class is allowed for business travel?")
    pipeline = RAGPipeline(retriever, llm)

    response = pipeline.ask_with_citations("And for flights?", history=HISTORY)

    assert retriever.queries == ["Which flight class is allowed for business travel?"]
    assert response.standalone_question == "Which flight class is allowed for business travel?"
    assert response.cited_source_ids == [1]


def test_no_history_means_no_extra_llm_call() -> None:
    llm = ScriptedLLM("unused")
    pipeline = RAGPipeline(RecordingRetriever(), llm)

    response = pipeline.ask_with_citations("Which flight class is allowed?")

    assert len(llm.prompts) == 1
    assert response.standalone_question is None


def test_failed_or_runaway_rewrite_falls_back_to_original_question() -> None:
    for bad_rewrite in (GenerationError("boom"), "", "x" * 2000):
        retriever = RecordingRetriever()
        pipeline = RAGPipeline(retriever, ScriptedLLM(bad_rewrite))

        pipeline.ask_with_citations("And for flights?", history=HISTORY)

        assert retriever.queries == ["And for flights?"]


def test_condense_prompt_keeps_only_recent_trimmed_turns() -> None:
    history = [("user", f"turn {i} " + "word " * 500) for i in range(10)]

    prompt = build_condense_question_prompt("And then?", history)

    assert "turn 3" not in prompt
    assert "turn 9" in prompt
    assert len(prompt) < 6 * 700 + 1000


class HistoryRecordingPipeline:
    def __init__(self) -> None:
        self.history = None

    def ask_with_citations(self, query, top_k=5, metadata_filters=None, history=None):
        self.history = history
        return RAGResponse(
            answer=ABSTENTION_ANSWER,
            sources=[],
            cited_source_ids=[],
            standalone_question="Rewritten?",
        )


def test_api_passes_history_and_returns_standalone_question(client: TestClient) -> None:
    pipeline = HistoryRecordingPipeline()
    app.dependency_overrides[get_rag_pipeline_provider] = lambda: (lambda: pipeline)

    response = client.post(
        "/qa",
        json={
            "question": "And for flights?",
            "history": [
                {"role": "user", "content": HISTORY[0][1]},
                {"role": "assistant", "content": HISTORY[1][1]},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["standalone_question"] == "Rewritten?"
    assert pipeline.history == HISTORY


def test_api_rejects_oversized_history(client: TestClient) -> None:
    history = [{"role": "user", "content": "hi"}] * 13

    response = client.post("/qa", json={"question": "Why?", "history": history})

    assert response.status_code == 422
