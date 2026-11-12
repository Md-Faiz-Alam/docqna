"""Post-generation grounding check.

Small local models sometimes answer questions the documents do not cover and
attach a citation anyway ("France won the 2018 World Cup [1]"). After citation
validation, the answer text is scored against the sources it cites with the
same cross-encoder used for reranking. If no cited source supports the answer
well enough, the pipeline abstains.

Scoring is sentence-level: a claim is supported by a specific sentence, and a
long chunk that mostly covers other topics would otherwise drag the score of a
correct answer down. Each cited source is split into sentences and pairs of
adjacent sentences, and the best match counts.

Calibrated in experiments/run_phase23_answerability_recalibration.py.
"""

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

_CITATION = re.compile(r"\s*\[\d+\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


class PairScorer(Protocol):
    """Anything with a CrossEncoder-style `predict(pairs)`."""

    # Typed loosely: CrossEncoder.predict accepts many input/return shapes.
    def predict(self, sentences: Any, /) -> Any: ...


@dataclass(frozen=True)
class GroundingDecision:
    is_grounded: bool
    score: float | None


def evidence_spans(text: str) -> list[str]:
    """Sentences of `text` plus each pair of adjacent sentences."""

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if len(s.strip()) > 3]
    pairs = [f"{a} {b}" for a, b in pairwise(sentences)]
    return sentences + pairs or [text]


class GroundingVerifier:
    def __init__(self, scorer: PairScorer, min_score: float) -> None:
        self.scorer = scorer
        self.min_score = min_score

    @staticmethod
    def strip_citations(answer: str) -> str:
        return _CITATION.sub("", answer).strip()

    def evaluate(self, answer: str, cited_texts: list[str]) -> GroundingDecision:
        claim = self.strip_citations(answer)
        if not claim or not cited_texts:
            return GroundingDecision(is_grounded=False, score=None)

        spans = list(dict.fromkeys(span for text in cited_texts for span in evidence_spans(text)))
        scores = self.scorer.predict([(claim, span) for span in spans])
        best = max(float(score) for score in scores)
        return GroundingDecision(is_grounded=best >= self.min_score, score=best)
