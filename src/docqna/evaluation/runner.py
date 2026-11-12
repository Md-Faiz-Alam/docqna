import json
import time
from pathlib import Path

from docqna.evaluation.answer_metrics import (
    answer_word_count,
    citation_presence,
    citation_validity,
    cited_evidence_text,
    matched_terms,
    term_coverage,
)
from docqna.evaluation.answerability_metrics import f1_score, response_abstained, safe_divide
from docqna.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    page_recall_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from docqna.evaluation.models import (
    AnswerabilityCaseResult,
    AnswerabilityEvaluationReport,
    AnswerCaseResult,
    AnswerEvaluationReport,
    EvaluationCase,
    RetrievalCaseResult,
    RetrievalEvaluationReport,
)
from docqna.exceptions import DocQNAError
from docqna.filters import MetadataFilters
from docqna.rag.pipeline import RAGPipeline
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.models import RetrievalResult


def load_evaluation_cases(path: str | Path) -> list[EvaluationCase]:
    """
    Load and validate evaluation cases from JSON.

    Both the legacy Phase 11 datasets and the Phase 21 v2 dataset
    are supported through ``EvaluationCase`` compatibility logic.
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset JSON must contain a list of cases.")

    if not data:
        raise ValueError("Evaluation dataset cannot be empty.")

    cases = [EvaluationCase.model_validate(item) for item in data]

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation dataset contains duplicate case_id values.")

    return cases


def _format_page_range(result: RetrievalResult) -> str:
    """Format page metadata for diagnostics."""

    if result.page_start is None:
        return "unknown"

    if result.page_end is None or result.page_end == result.page_start:
        return str(result.page_start)

    return f"{result.page_start}-{result.page_end}"


class RetrievalEvaluator:
    """Evaluate a retriever against labeled questions."""

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def evaluate(self, cases: list[EvaluationCase], top_k: int = 5) -> RetrievalEvaluationReport:
        """Run retrieval evaluation."""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        if top_k < 5:
            raise ValueError("top_k must be at least 5 for baseline evaluation.")

        answerable_cases = [case for case in cases if case.should_answer]

        if not answerable_cases:
            raise ValueError("Retrieval evaluation requires at least one answerable case.")

        num_skipped_unanswerable = len(cases) - len(answerable_cases)

        case_results: list[RetrievalCaseResult] = []

        for case in answerable_cases:
            results = self.retriever.retrieve(query=case.question, top_k=top_k)

            relevant_pages = set(case.relevant_pages)

            relevant_chunk_ids = set(case.relevant_chunk_ids)

            case_results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    hit_at_1=hit_at_k(results, relevant_pages, 1, relevant_chunk_ids),
                    hit_at_3=hit_at_k(results, relevant_pages, 3, relevant_chunk_ids),
                    hit_at_5=hit_at_k(results, relevant_pages, 5, relevant_chunk_ids),
                    reciprocal_rank=reciprocal_rank(results, relevant_pages, 5, relevant_chunk_ids),
                    page_recall_at_5=page_recall_at_k(results, relevant_pages, 5),
                    precision_at_k=precision_at_k(
                        results, relevant_pages, top_k, relevant_chunk_ids
                    ),
                    recall_at_k=recall_at_k(results, relevant_pages, top_k, relevant_chunk_ids),
                    ndcg_at_k=ndcg_at_k(results, relevant_pages, top_k, relevant_chunk_ids),
                    retrieved_chunk_ids=[result.chunk_id for result in results],
                    retrieved_pages=[_format_page_range(result) for result in results],
                    distances=[result.distance for result in results],
                )
            )

        num_cases = len(case_results)

        return RetrievalEvaluationReport(
            num_cases=num_cases,
            top_k=top_k,
            num_skipped_unanswerable=(num_skipped_unanswerable),
            mean_hit_at_1=sum(result.hit_at_1 for result in case_results) / num_cases,
            mean_hit_at_3=sum(result.hit_at_3 for result in case_results) / num_cases,
            mean_hit_at_5=sum(result.hit_at_5 for result in case_results) / num_cases,
            mean_reciprocal_rank=sum(result.reciprocal_rank for result in case_results) / num_cases,
            mean_page_recall_at_5=sum(result.page_recall_at_5 for result in case_results)
            / num_cases,
            mean_precision_at_k=sum(result.precision_at_k for result in case_results) / num_cases,
            mean_recall_at_k=sum(result.recall_at_k for result in case_results) / num_cases,
            mean_ndcg_at_k=sum(result.ndcg_at_k for result in case_results) / num_cases,
            cases=case_results,
        )


class AnswerEvaluator:
    """
    Evaluate generated RAG answers and citations.

    Supports checkpointing and resume.
    """

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> list[AnswerCaseResult]:
        """Load existing checkpoint results."""

        if not checkpoint_path.exists():
            return []

        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))

            return [AnswerCaseResult.model_validate(item) for item in data]

        except (json.JSONDecodeError, OSError, ValueError) as exc:
            raise ValueError(f"Could not load checkpoint: {checkpoint_path}") from exc

    @staticmethod
    def _save_checkpoint(checkpoint_path: Path, results: list[AnswerCaseResult]) -> None:
        """Save completed evaluation cases."""

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        data = [result.model_dump() for result in results]

        checkpoint_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
        checkpoint_path: str | Path | None = None,
        resume: bool = True,
    ) -> AnswerEvaluationReport:
        """Run answer and citation evaluation."""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        checkpoint: Path | None = None

        previous_results: list[AnswerCaseResult] = []

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)

            if resume:
                previous_results = self._load_checkpoint(checkpoint)

        successful_previous = {
            result.case_id: result for result in previous_results if result.error is None
        }

        case_results: list[AnswerCaseResult] = []

        for index, case in enumerate(cases, start=1):
            previous = successful_previous.get(case.case_id)

            if previous is not None:
                print(f"[{index}/{len(cases)}] {case.case_id} - resumed")

                case_results.append(previous)

                continue

            print(f"[{index}/{len(cases)}] {case.case_id} - generating...")

            started = time.perf_counter()

            try:
                response = self.rag_pipeline.ask_with_citations(query=case.question, top_k=top_k)

                elapsed = time.perf_counter() - started

                answer_matches = matched_terms(response.answer, case.expected_answer_terms)

                evidence_text = cited_evidence_text(response)

                evidence_matches = matched_terms(evidence_text, case.expected_answer_terms)

                case_result = AnswerCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer=response.answer,
                    expected_answer_terms=(case.expected_answer_terms),
                    matched_answer_terms=(answer_matches),
                    answer_term_coverage=(
                        term_coverage(response.answer, case.expected_answer_terms)
                    ),
                    citation_presence=(citation_presence(response)),
                    citation_validity=(citation_validity(response)),
                    cited_source_ids=(response.cited_source_ids),
                    matched_cited_evidence_terms=(evidence_matches),
                    cited_evidence_term_coverage=(
                        term_coverage(evidence_text, case.expected_answer_terms)
                    ),
                    answer_word_count=(answer_word_count(response.answer)),
                    elapsed_seconds=elapsed,
                    error=None,
                )

                print(f"    completed in {elapsed:.2f}s")

            except DocQNAError as exc:
                elapsed = time.perf_counter() - started

                case_result = AnswerCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    answer="",
                    expected_answer_terms=(case.expected_answer_terms),
                    matched_answer_terms=[],
                    answer_term_coverage=0.0,
                    citation_presence=0.0,
                    citation_validity=0.0,
                    cited_source_ids=[],
                    matched_cited_evidence_terms=[],
                    cited_evidence_term_coverage=0.0,
                    answer_word_count=0,
                    elapsed_seconds=elapsed,
                    error=(f"{type(exc).__name__}: {exc}"),
                )

                print(f"    failed after {elapsed:.2f}s")

                print("    error:", case_result.error)

            case_results.append(case_result)

            if checkpoint is not None:
                self._save_checkpoint(checkpoint, case_results)

                print("    checkpoint saved")

        num_cases = len(case_results)

        successful_cases = sum(1 for result in case_results if result.error is None)

        failed_cases = num_cases - successful_cases

        total_case_elapsed_seconds = sum(result.elapsed_seconds for result in case_results)

        mean_case_elapsed_seconds = total_case_elapsed_seconds / num_cases

        return AnswerEvaluationReport(
            num_cases=num_cases,
            successful_cases=successful_cases,
            failed_cases=failed_cases,
            top_k=top_k,
            mean_answer_term_coverage=sum(result.answer_term_coverage for result in case_results)
            / num_cases,
            mean_citation_presence=sum(result.citation_presence for result in case_results)
            / num_cases,
            mean_citation_validity=sum(result.citation_validity for result in case_results)
            / num_cases,
            mean_cited_evidence_term_coverage=sum(
                result.cited_evidence_term_coverage for result in case_results
            )
            / num_cases,
            mean_answer_word_count=sum(result.answer_word_count for result in case_results)
            / num_cases,
            mean_case_elapsed_seconds=(mean_case_elapsed_seconds),
            total_case_elapsed_seconds=(total_case_elapsed_seconds),
            cases=case_results,
        )


class AnswerabilityEvaluator:
    """
    Evaluate whether the RAG system answers or abstains correctly.

    Positive-class metrics named ``answer_*`` treat "should answer"
    as the positive class. ``abstention_*`` metrics treat "should
    abstain" as the positive class.
    """

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> AnswerabilityEvaluationReport:
        """Run answerability and abstention evaluation."""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        case_results: list[AnswerabilityCaseResult] = []

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id} - evaluating answerability...")

            started = time.perf_counter()

            try:
                if metadata_filters:
                    response = self.rag_pipeline.ask_with_citations(
                        query=case.question, top_k=top_k, metadata_filters=metadata_filters
                    )
                else:
                    response = self.rag_pipeline.ask_with_citations(
                        query=case.question, top_k=top_k
                    )

                elapsed = time.perf_counter() - started

                abstained = response_abstained(response)

                predicted_should_answer = not abstained

                correct = predicted_should_answer == case.should_answer

                case_result = AnswerabilityCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    should_answer=(case.should_answer),
                    predicted_should_answer=(predicted_should_answer),
                    abstained=abstained,
                    correct=correct,
                    elapsed_seconds=elapsed,
                    error=None,
                )

            except DocQNAError as exc:
                elapsed = time.perf_counter() - started

                case_result = AnswerabilityCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    should_answer=(case.should_answer),
                    predicted_should_answer=None,
                    abstained=None,
                    correct=False,
                    elapsed_seconds=elapsed,
                    error=(f"{type(exc).__name__}: {exc}"),
                )

            case_results.append(case_result)

        successful_results = [result for result in case_results if result.error is None]

        successful_cases = len(successful_results)

        error_cases = len(case_results) - successful_cases

        true_answers = sum(
            1
            for result in successful_results
            if (result.should_answer and result.predicted_should_answer is True)
        )

        true_abstentions = sum(
            1
            for result in successful_results
            if (not result.should_answer and result.abstained is True)
        )

        false_answers = sum(
            1
            for result in successful_results
            if (not result.should_answer and result.predicted_should_answer is True)
        )

        false_abstentions = sum(
            1
            for result in successful_results
            if (result.should_answer and result.abstained is True)
        )

        accuracy = safe_divide(true_answers + true_abstentions, successful_cases)

        answer_precision = safe_divide(true_answers, true_answers + false_answers)

        answer_recall = safe_divide(true_answers, true_answers + false_abstentions)

        answer_f1 = f1_score(answer_precision, answer_recall)

        abstention_precision = safe_divide(true_abstentions, true_abstentions + false_abstentions)

        abstention_recall = safe_divide(true_abstentions, true_abstentions + false_answers)

        abstention_f1 = f1_score(abstention_precision, abstention_recall)

        total_case_elapsed_seconds = sum(result.elapsed_seconds for result in case_results)

        mean_case_elapsed_seconds = total_case_elapsed_seconds / len(case_results)

        return AnswerabilityEvaluationReport(
            num_cases=len(case_results),
            successful_cases=successful_cases,
            error_cases=error_cases,
            top_k=top_k,
            true_answers=true_answers,
            true_abstentions=true_abstentions,
            false_answers=false_answers,
            false_abstentions=false_abstentions,
            accuracy=accuracy,
            answer_precision=answer_precision,
            answer_recall=answer_recall,
            answer_f1=answer_f1,
            abstention_precision=(abstention_precision),
            abstention_recall=(abstention_recall),
            abstention_f1=abstention_f1,
            mean_case_elapsed_seconds=(mean_case_elapsed_seconds),
            total_case_elapsed_seconds=(total_case_elapsed_seconds),
            cases=case_results,
        )
