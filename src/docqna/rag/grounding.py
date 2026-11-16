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

Calibrated in experiments/calibrate_answerability.py.
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
    def __init__(self, scorer: PairScorer, min_score: float, per_sentence: bool = False) -> None:
        self.scorer = scorer
        self.min_score = min_score
        # Production turns this on (GROUNDING_PER_SENTENCE); the default keeps
        # the whole-answer score the earlier calibration experiments used.
        self.per_sentence = per_sentence

    def check(self, answer: str, sources: list[str], citation_ids: list[int]) -> GroundingDecision:
        """Run the configured check. `sources[i]` is the text behind citation `[i + 1]`."""

        if self.per_sentence:
            return self.evaluate_claims(answer, sources)
        cited = [sources[cid - 1] for cid in citation_ids if 1 <= cid <= len(sources)]
        return self.evaluate(answer, cited)

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

    def evaluate_claims(self, answer: str, sources: list[str]) -> GroundingDecision:
        """Every claim must be supported by a source it cites.

        `sources[i]` is the text behind citation `[i + 1]`. Each sentence of
        the answer is scored only against the sources that sentence cites,
        and the answer's score is its weakest sentence. A made-up sentence
        therefore fails the answer even when another sentence is well
        supported, and a long answer is no longer scored as one blurred claim.
        """

        claims = answer_claims(answer)
        if not claims:
            return GroundingDecision(is_grounded=False, score=None)

        pairs: list[tuple[str, str]] = []
        owners: list[int] = []
        for index, (claim, citation_ids) in enumerate(claims):
            texts = [sources[cid - 1] for cid in citation_ids if 1 <= cid <= len(sources)]
            spans = dict.fromkeys(span for text in texts for span in evidence_spans(text))
            if not spans:
                return GroundingDecision(is_grounded=False, score=None)
            pairs.extend((claim, span) for span in spans)
            owners.extend([index] * len(spans))

        best = [float("-inf")] * len(claims)
        for owner, score in zip(owners, self.scorer.predict(pairs), strict=True):
            best[owner] = max(best[owner], float(score))
        weakest = min(best)
        return GroundingDecision(is_grounded=weakest >= self.min_score, score=weakest)


_CLAIM_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\s\[])|\n+")
_CITATION_ID = re.compile(r"\[(\d+)\]")


def answer_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into (sentence without citations, citation ids) pairs.

    A sentence without its own citation is covered by the citations of the
    next cited sentence ("A. B [1]." -> both cite [1]); trailing uncited
    sentences take the last citations seen. Citations placed after the full
    stop ("A. [1]") stay with their sentence.
    """

    claims: list[tuple[str, list[int]]] = []
    pending: list[str] = []
    for part in _CLAIM_SPLIT.split(answer.strip()):
        text = _CITATION.sub("", part).strip()
        ids = list(dict.fromkeys(int(n) for n in _CITATION_ID.findall(part)))
        if len(text) <= 3:
            text = ""
        if not ids:
            if text:
                pending.append(text)
            continue
        claims.extend((sentence, ids) for sentence in pending)
        pending = []
        if text:
            claims.append((text, ids))
        elif claims:
            # A bare "[2]" after the sentence it belongs to.
            last_text, last_ids = claims[-1]
            claims[-1] = (last_text, list(dict.fromkeys([*last_ids, *ids])))
    if pending and claims:
        claims.extend((sentence, claims[-1][1]) for sentence in pending)
    return claims
