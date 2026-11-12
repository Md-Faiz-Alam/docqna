import json
import time

from pydantic import ValidationError

from docqna.evaluation.answerability_metrics import response_abstained
from docqna.evaluation.models import (
    EvaluationCase,
    RAGQualityCaseResult,
    RAGQualityEvaluationReport,
    RAGQualityScores,
)
from docqna.exceptions import DocQNAError
from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.models import RAGResponse
from docqna.rag.pipeline import RAGPipeline

JUDGE_SYSTEM_PROMPT = """
You are a strict evaluator for a retrieval-augmented generation
(RAG) system.

Evaluate ONLY from the supplied question, generated answer, and
retrieved context. Do not use outside knowledge.

Return exactly one JSON object and no markdown.

Use scores from 0.0 to 1.0:

answer_relevance:
How directly and completely the answer addresses the question.

context_relevance:
How relevant the retrieved context is to answering the question.

faithfulness:
How fully the factual claims in the answer are supported by the
retrieved context. Unsupported or contradictory claims lower this
score.

citation_correctness:
How well the cited source IDs support the claims for which they are
used. A citation is not correct merely because its numeric ID is
valid.

Also return one short rationale explaining the scores.
""".strip()


def _format_sources(response: RAGResponse) -> str:
    """Format retrieved sources for judge input."""

    blocks: list[str] = []

    cited_ids = set(response.cited_source_ids)

    for source in response.sources:
        cited_marker = "yes" if source.citation_id in cited_ids else "no"

        page_text = "unknown"

        if source.page_start is not None:
            page_text = str(source.page_start)

            if source.page_end is not None and source.page_end != source.page_start:
                page_text = f"{source.page_start}-{source.page_end}"

        blocks.append(
            "\n".join(
                [
                    (f"[Source {source.citation_id}] cited={cited_marker}"),
                    (f"file={source.filename} pages={page_text}"),
                    source.text.strip(),
                ]
            )
        )

    return "\n\n".join(blocks)


def build_quality_judge_prompt(*, question: str, response: RAGResponse) -> str:
    """Build one deterministic RAG-quality judge prompt."""

    sources_text = _format_sources(response)

    if not sources_text:
        sources_text = "(no retrieved sources)"

    return f"""
Evaluate this RAG response.

QUESTION:
{question}

GENERATED ANSWER:
{response.answer}

RETRIEVED CONTEXT:
{sources_text}

Return JSON with exactly these keys:
{{
  "answer_relevance": 0.0,
  "context_relevance": 0.0,
  "faithfulness": 0.0,
  "citation_correctness": 0.0,
  "rationale": "short explanation"
}}
""".strip()


def _extract_json_object(text: str) -> dict[str, object]:
    """
    Parse a JSON object from a judge response.

    The direct parse is preferred. A bounded fallback extracts the
    first outer JSON object if a local model adds surrounding text.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    try:
        payload = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end < 0 or end <= start:
            raise ValueError("RAG quality judge did not return a JSON object.") from None

        try:
            payload = json.loads(cleaned[start : end + 1])

        except json.JSONDecodeError as exc:
            raise ValueError("RAG quality judge returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("RAG quality judge JSON must be an object.")

    return payload


class RAGQualityJudge:
    """LLM-as-judge scorer using the existing provider interface."""

    def __init__(self, llm: BaseLLMProvider) -> None:
        self.llm = llm

    def score(self, *, question: str, response: RAGResponse) -> RAGQualityScores:
        """Score one answered RAG response."""

        prompt = build_quality_judge_prompt(question=question, response=response)

        raw = self.llm.generate(prompt=prompt, system_prompt=(JUDGE_SYSTEM_PROMPT))

        payload = _extract_json_object(raw)

        try:
            return RAGQualityScores.model_validate(payload)

        except ValidationError as exc:
            raise ValueError("RAG quality judge returned invalid score fields.") from exc


class RAGQualityEvaluator:
    """
    Evaluate answer relevance, context relevance, faithfulness,
    and semantic citation correctness.

    Unanswerable cases are intentionally skipped here because
    abstention behavior is measured separately by Phase 21C.
    """

    def __init__(self, rag_pipeline: RAGPipeline, judge: RAGQualityJudge) -> None:
        self.rag_pipeline = rag_pipeline
        self.judge = judge

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> RAGQualityEvaluationReport:
        """Run RAG-quality evaluation."""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        case_results: list[RAGQualityCaseResult] = []

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id} - evaluating RAG quality...")

            if not case.should_answer:
                case_results.append(
                    RAGQualityCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        should_answer=False,
                        status=("skipped_unanswerable"),
                    )
                )

                print("    skipped (unanswerable case)")

                continue

            total_started = time.perf_counter()

            generation_started = time.perf_counter()

            try:
                if metadata_filters:
                    response = self.rag_pipeline.ask_with_citations(
                        query=case.question, top_k=top_k, metadata_filters=metadata_filters
                    )
                else:
                    response = self.rag_pipeline.ask_with_citations(
                        query=case.question, top_k=top_k
                    )

                generation_elapsed = time.perf_counter() - generation_started

                if response_abstained(response):
                    total_elapsed = time.perf_counter() - total_started

                    case_results.append(
                        RAGQualityCaseResult(
                            case_id=case.case_id,
                            question=case.question,
                            should_answer=True,
                            status="abstained",
                            answer=response.answer,
                            cited_source_ids=(response.cited_source_ids),
                            source_count=len(response.sources),
                            generation_elapsed_seconds=(generation_elapsed),
                            judge_elapsed_seconds=0.0,
                            total_elapsed_seconds=(total_elapsed),
                        )
                    )

                    print("    not judged (pipeline abstained)")

                    continue

                judge_started = time.perf_counter()

                scores = self.judge.score(question=case.question, response=response)

                judge_elapsed = time.perf_counter() - judge_started

                total_elapsed = time.perf_counter() - total_started

                case_results.append(
                    RAGQualityCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        should_answer=True,
                        status="judged",
                        answer=response.answer,
                        answer_relevance=(scores.answer_relevance),
                        context_relevance=(scores.context_relevance),
                        faithfulness=(scores.faithfulness),
                        citation_correctness=(scores.citation_correctness),
                        rationale=(scores.rationale),
                        cited_source_ids=(response.cited_source_ids),
                        source_count=len(response.sources),
                        generation_elapsed_seconds=(generation_elapsed),
                        judge_elapsed_seconds=(judge_elapsed),
                        total_elapsed_seconds=(total_elapsed),
                    )
                )

                print(f"    judged in {total_elapsed:.2f}s")

            except (DocQNAError, ValueError) as exc:
                generation_elapsed = time.perf_counter() - generation_started

                total_elapsed = time.perf_counter() - total_started

                case_results.append(
                    RAGQualityCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        should_answer=True,
                        status="error",
                        generation_elapsed_seconds=(generation_elapsed),
                        total_elapsed_seconds=(total_elapsed),
                        error=(f"{type(exc).__name__}: {exc}"),
                    )
                )

                print(f"    failed: {type(exc).__name__}")

        judged_results = [result for result in case_results if result.status == "judged"]

        answerable_cases = sum(1 for result in case_results if result.should_answer)

        skipped_unanswerable_cases = sum(
            1 for result in case_results if (result.status == "skipped_unanswerable")
        )

        abstained_cases = sum(1 for result in case_results if result.status == "abstained")

        failed_cases = sum(1 for result in case_results if result.status == "error")

        judged_cases = len(judged_results)

        def mean_score(attribute: str) -> float:
            if not judged_results:
                return 0.0

            values = [getattr(result, attribute) for result in judged_results]

            numeric_values = [value for value in values if value is not None]

            if not numeric_values:
                return 0.0

            return sum(numeric_values) / len(numeric_values)

        evaluated_results = [result for result in case_results if result.should_answer]

        total_elapsed_seconds = sum(result.total_elapsed_seconds for result in evaluated_results)

        mean_generation_elapsed_seconds = (
            sum(result.generation_elapsed_seconds for result in evaluated_results)
            / len(evaluated_results)
            if evaluated_results
            else 0.0
        )

        mean_judge_elapsed_seconds = (
            sum(result.judge_elapsed_seconds for result in judged_results) / judged_cases
            if judged_cases
            else 0.0
        )

        mean_total_elapsed_seconds = (
            total_elapsed_seconds / len(evaluated_results) if evaluated_results else 0.0
        )

        return RAGQualityEvaluationReport(
            num_cases=len(case_results),
            answerable_cases=(answerable_cases),
            skipped_unanswerable_cases=(skipped_unanswerable_cases),
            judged_cases=judged_cases,
            abstained_cases=abstained_cases,
            failed_cases=failed_cases,
            top_k=top_k,
            mean_answer_relevance=(mean_score("answer_relevance")),
            mean_context_relevance=(mean_score("context_relevance")),
            mean_faithfulness=(mean_score("faithfulness")),
            mean_citation_correctness=(mean_score("citation_correctness")),
            mean_generation_elapsed_seconds=(mean_generation_elapsed_seconds),
            mean_judge_elapsed_seconds=(mean_judge_elapsed_seconds),
            mean_total_elapsed_seconds=(mean_total_elapsed_seconds),
            total_elapsed_seconds=(total_elapsed_seconds),
            cases=case_results,
        )
