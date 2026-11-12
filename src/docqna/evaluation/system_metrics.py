import time
from typing import Any, cast

from docqna.evaluation.answerability_metrics import response_abstained
from docqna.evaluation.models import (
    EvaluationCase,
    SystemEvaluationCaseResult,
    SystemEvaluationReport,
)
from docqna.exceptions import DocQNAError
from docqna.filters import MetadataFilters
from docqna.generation.providers.base import BaseLLMProvider
from docqna.rag.pipeline import RAGPipeline
from docqna.retrieval.base import BaseRetriever


class _TimedRetriever:
    """
    Transparent retrieval proxy used only for offline evaluation.

    It delegates to the production retriever while measuring the
    actual time spent inside ``retrieve``.
    """

    def __init__(self, delegate: BaseRetriever) -> None:
        self.delegate = delegate
        self.elapsed_seconds = 0.0

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()

        try:
            return self.delegate.retrieve(*args, **kwargs)

        finally:
            self.elapsed_seconds += time.perf_counter() - started


class _TimedLLMProvider(BaseLLMProvider):
    """Transparent LLM proxy that records generation latency."""

    def __init__(self, delegate: BaseLLMProvider) -> None:
        self.delegate = delegate
        self.elapsed_seconds = 0.0

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        started = time.perf_counter()

        try:
            return self.delegate.generate(prompt=prompt, system_prompt=system_prompt)

        finally:
            self.elapsed_seconds += time.perf_counter() - started

    def health_check(self) -> bool:
        return self.delegate.health_check()


class SystemEvaluator:
    """
    Measure end-to-end RAG latency and reliability.

    The evaluator runs the real ``RAGPipeline.ask_with_citations``
    path through transparent timing proxies. This avoids changing
    production response models or duplicating RAG logic.
    """

    def __init__(self, rag_pipeline: RAGPipeline) -> None:
        self.rag_pipeline = rag_pipeline

    def _build_instrumented_pipeline(
        self,
    ) -> tuple[RAGPipeline, _TimedRetriever, _TimedLLMProvider]:
        timed_retriever = _TimedRetriever(self.rag_pipeline.retriever)

        timed_llm = _TimedLLMProvider(self.rag_pipeline.llm)

        instrumented = RAGPipeline(
            retriever=cast(BaseRetriever, timed_retriever),
            llm=timed_llm,
            confidence_gate=(self.rag_pipeline.confidence_gate),
        )

        return (instrumented, timed_retriever, timed_llm)

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int = 5,
        metadata_filters: MetadataFilters | None = None,
    ) -> SystemEvaluationReport:
        """Run system-level latency and reliability evaluation."""

        if not cases:
            raise ValueError("Evaluation cases cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        case_results: list[SystemEvaluationCaseResult] = []

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case.case_id} - measuring system performance...")

            (instrumented, timed_retriever, timed_llm) = self._build_instrumented_pipeline()

            total_started = time.perf_counter()

            try:
                if metadata_filters:
                    response = instrumented.ask_with_citations(
                        query=case.question, top_k=top_k, metadata_filters=metadata_filters
                    )
                else:
                    response = instrumented.ask_with_citations(query=case.question, top_k=top_k)

                total_elapsed = time.perf_counter() - total_started

                retrieval_elapsed = timed_retriever.elapsed_seconds

                generation_elapsed = timed_llm.elapsed_seconds

                postprocessing_elapsed = max(
                    0.0, total_elapsed - retrieval_elapsed - generation_elapsed
                )

                abstained = response_abstained(response)

                case_results.append(
                    SystemEvaluationCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        should_answer=(case.should_answer),
                        success=True,
                        abstained=abstained,
                        retrieval_elapsed_seconds=(retrieval_elapsed),
                        generation_elapsed_seconds=(generation_elapsed),
                        postprocessing_elapsed_seconds=(postprocessing_elapsed),
                        total_elapsed_seconds=(total_elapsed),
                        source_count=len(response.sources),
                        cited_source_count=len(response.cited_source_ids),
                        error=None,
                    )
                )

                print(
                    "    success "
                    f"total={total_elapsed:.3f}s "
                    f"retrieval={retrieval_elapsed:.3f}s "
                    f"generation={generation_elapsed:.3f}s"
                )

            except (DocQNAError, ValueError) as exc:
                total_elapsed = time.perf_counter() - total_started

                retrieval_elapsed = timed_retriever.elapsed_seconds

                generation_elapsed = timed_llm.elapsed_seconds

                postprocessing_elapsed = max(
                    0.0, total_elapsed - retrieval_elapsed - generation_elapsed
                )

                case_results.append(
                    SystemEvaluationCaseResult(
                        case_id=case.case_id,
                        question=case.question,
                        should_answer=(case.should_answer),
                        success=False,
                        abstained=None,
                        retrieval_elapsed_seconds=(retrieval_elapsed),
                        generation_elapsed_seconds=(generation_elapsed),
                        postprocessing_elapsed_seconds=(postprocessing_elapsed),
                        total_elapsed_seconds=(total_elapsed),
                        error=(f"{type(exc).__name__}: {exc}"),
                    )
                )

                print(f"    failed: {type(exc).__name__}")

        successful_results = [result for result in case_results if result.success]

        successful_cases = len(successful_results)

        failed_cases = len(case_results) - successful_cases

        answered_cases = sum(1 for result in successful_results if result.abstained is False)

        abstained_cases = sum(1 for result in successful_results if result.abstained is True)

        num_cases = len(case_results)

        success_rate = successful_cases / num_cases

        error_rate = failed_cases / num_cases

        abstention_rate = abstained_cases / successful_cases if successful_cases else 0.0

        def mean(values: list[float]) -> float:
            if not values:
                return 0.0

            return sum(values) / len(values)

        total_elapsed_seconds = sum(result.total_elapsed_seconds for result in case_results)

        return SystemEvaluationReport(
            num_cases=num_cases,
            successful_cases=(successful_cases),
            failed_cases=failed_cases,
            answered_cases=answered_cases,
            abstained_cases=abstained_cases,
            top_k=top_k,
            success_rate=success_rate,
            error_rate=error_rate,
            abstention_rate=(abstention_rate),
            mean_retrieval_elapsed_seconds=mean(
                [result.retrieval_elapsed_seconds for result in case_results]
            ),
            mean_generation_elapsed_seconds=mean(
                [result.generation_elapsed_seconds for result in case_results]
            ),
            mean_postprocessing_elapsed_seconds=mean(
                [result.postprocessing_elapsed_seconds for result in case_results]
            ),
            mean_total_elapsed_seconds=mean(
                [result.total_elapsed_seconds for result in case_results]
            ),
            total_elapsed_seconds=(total_elapsed_seconds),
            cases=case_results,
        )
