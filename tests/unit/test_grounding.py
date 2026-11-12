from collections.abc import Iterator

from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline, RAGResponse
from docqna.rag.grounding import GroundingVerifier
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult

POLICY = "Meal expenses are capped at 60 euros per day while travelling."


class OverlapScorer:
    """Fake cross-encoder: +5 if the claim shares a number with the source, else -10."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        return [5.0 if "60" in claim and "60" in source else -10.0 for claim, source in pairs]


class OneSource(BaseRetriever):
    def retrieve(self, query, top_k=5, metadata_filters=None):
        return [
            RetrievalResult(
                chunk_id="c1",
                text=POLICY,
                rank=1,
                distance=-2.0,
                score=2.0,
                filename="handbook.txt",
                file_type="txt",
                chunk_index=0,
            )
        ]


class FixedLLM(BaseLLMProvider):
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.answer

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        yield self.answer

    def health_check(self) -> bool:
        return True


def test_verifier_scores_claim_without_citation_markers() -> None:
    scorer = OverlapScorer()
    verifier = GroundingVerifier(scorer, min_score=0.7)

    decision = verifier.evaluate("The cap is 60 euros [1].", [POLICY])

    assert decision.is_grounded
    assert decision.score == 5.0
    assert scorer.pairs == [("The cap is 60 euros.", POLICY)]


def test_verifier_rejects_empty_claims_and_missing_sources() -> None:
    verifier = GroundingVerifier(OverlapScorer(), min_score=0.7)

    assert not verifier.evaluate("[1]", [POLICY]).is_grounded
    assert not verifier.evaluate("The cap is 60 euros.", []).is_grounded


def make_pipeline(answer: str) -> RAGPipeline:
    return RAGPipeline(
        OneSource(),
        FixedLLM(answer),
        grounding_verifier=GroundingVerifier(OverlapScorer(), min_score=0.7),
    )


def test_supported_answer_is_returned() -> None:
    response = make_pipeline("Meals are capped at 60 euros per day [1].").ask_with_citations(
        "What is the meal cap?"
    )

    assert response.answer == "Meals are capped at 60 euros per day [1]."


def test_cited_but_unsupported_answer_becomes_abstention() -> None:
    response = make_pipeline("France won the 2018 World Cup [1].").ask_with_citations(
        "Who won the 2018 World Cup?"
    )

    assert response.answer == ABSTENTION_ANSWER
    assert response.sources == []


def test_streaming_final_answer_is_also_grounding_checked() -> None:
    items = list(
        make_pipeline("France won the 2018 World Cup [1].").stream_with_citations("Who won?")
    )

    final = items[-1]
    assert isinstance(final, RAGResponse)
    assert final.answer == ABSTENTION_ANSWER
