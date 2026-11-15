"""How fast is DocQ&A, and what does an answer cost?

Measures the production pipeline (hybrid search, reranker, retrieval gate,
injection filter, LLM, grounding check) stage by stage:

- startup: loading the embedding and reranker models (a cold start)
- indexing: reading, chunking and embedding each document
- answering, per question, with streaming as in the web demo:
  search (dense + keyword), reranking, time until the first word appears,
  LLM generation, grounding check, and the total; plus the tokens the LLM used.

Cost is computed from the token counts and the provider's published prices
(`PRICES` below, with source and date). Local Ollama has no per-token price.

    python -m experiments.measure_speed_and_cost run --provider gemini
    python -m experiments.measure_speed_and_cost run --provider ollama
    python -m experiments.measure_speed_and_cost analyze

`--include-private` also times indexing of `data/samples/sample.pdf` (only
timings and counts are recorded, never its text). Results go to
`data/evaluation/results/speed_and_cost/`.
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docqna.api.dependencies import build_llm, build_session_retriever
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.exceptions import LLMConnectionError, LLMRateLimitError
from docqna.generation.providers.base import BaseLLMProvider, TokenUsage
from docqna.indexing import ChromaVectorStore, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRegistry
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline, RAGResponse
from docqna.rag.grounding import GroundingDecision, GroundingVerifier
from docqna.retrieval import CrossEncoderReranker
from docqna.retrieval.base import BaseRetriever
from docqna.retrieval.confidence import RetrievalConfidenceGate
from docqna.retrieval.models import RetrievalResult
from experiments.calibrate_answerability import (
    GEMINI_MIN_INTERVAL_SECONDS,
    PROVIDERS,
    RETRY_WAITS_SECONDS,
    SESSION_ID,
    TOP_K,
    experiment_settings,
    public_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results" / "speed_and_cost"

# Paid-tier list prices in US dollars per 1M tokens. Thinking tokens are billed
# as output. Checked on the date below; update both together.
PRICES: dict[str, dict[str, Any]] = {
    "gemini-3.5-flash-lite": {
        "input_per_million": 0.30,
        "output_per_million": 2.50,
        "source": "https://aistudio.google.com/docs/pricing",
        "checked": "2026-11-15",
        "free_tier": "Free of charge within rate limits; Google may use free-tier data "
        "to improve its products.",
    }
}

STAGES = ("search_ms", "rerank_ms", "first_token_ms", "generation_ms", "grounding_ms", "total_ms")


class Stopwatch:
    """Accumulates milliseconds per named stage."""

    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def add(self, stage: str, started: float) -> None:
        self.ms[stage] = self.ms.get(stage, 0.0) + (time.perf_counter() - started) * 1000.0


class TimedRetriever(BaseRetriever):
    def __init__(self, inner: BaseRetriever, watch: Stopwatch, stage: str) -> None:
        self.inner, self.watch, self.stage = inner, watch, stage

    def retrieve(
        self, query: str, top_k: int = 5, metadata_filters: dict[str, str | int] | None = None
    ) -> list[RetrievalResult]:
        started = time.perf_counter()
        try:
            return self.inner.retrieve(query, top_k=top_k, metadata_filters=metadata_filters)
        finally:
            self.watch.add(self.stage, started)


class TimedLLM(BaseLLMProvider):
    """Times every call and adds up the tokens the provider reports."""

    def __init__(self, inner: BaseLLMProvider, watch: Stopwatch) -> None:
        self.inner, self.watch = inner, watch
        self.calls = 0
        self.usage: list[TokenUsage] = []

    def _record(self) -> None:
        self.calls += 1
        if self.inner.last_usage is not None:
            self.usage.append(self.inner.last_usage)

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        started = time.perf_counter()
        try:
            return self.inner.generate(prompt, system_prompt=system_prompt)
        finally:
            self.watch.add("generation_ms", started)
            self._record()

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        started = time.perf_counter()
        try:
            yield from self.inner.generate_stream(prompt, system_prompt=system_prompt)
        finally:
            self.watch.add("generation_ms", started)
            self._record()

    def health_check(self) -> bool:
        return self.inner.health_check()


class TimedVerifier(GroundingVerifier):
    def __init__(self, inner: GroundingVerifier, watch: Stopwatch) -> None:
        super().__init__(scorer=inner.scorer, min_score=inner.min_score)
        self.watch = watch

    def evaluate(self, answer: str, cited_texts: list[str]) -> GroundingDecision:
        started = time.perf_counter()
        try:
            return super().evaluate(answer, cited_texts)
        finally:
            self.watch.add("grounding_ms", started)


def hardware() -> dict[str, Any]:
    return {
        "system": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def cost(usage: dict[str, float], model: str) -> float | None:
    """US dollars for one question's tokens at paid-tier prices, if known."""

    price = PRICES.get(model)
    if price is None:
        return None
    billed_output = usage["output_tokens"] + usage["thinking_tokens"]
    return (
        usage["input_tokens"] * price["input_per_million"]
        + billed_output * price["output_per_million"]
    ) / 1_000_000


def run(provider: str, ollama_url: str, include_private: bool) -> Path:
    settings = experiment_settings(provider, ollama_url)
    folder = "phi3_mini" if provider == "ollama" else provider
    out_dir = RESULTS_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-speed-"))
    documents, cases = public_cases()
    try:
        started = time.perf_counter()
        embedding = SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        embedding.embed_query("warm up")
        embedding_load_s = time.perf_counter() - started

        store = ChromaVectorStore(persist_directory=work_dir / "chroma", collection_name="speed")
        indexer = DocumentIndexService(
            ingestion_service=DocumentIngestionService(),
            registry=DocumentRegistry(work_dir / "registry.db"),
            embedding_model=embedding,
            vector_store=store,
        )
        indexing = []
        private = PROJECT_ROOT / "data/samples/sample.pdf"
        to_index = [*documents, *([private] if include_private and private.exists() else [])]
        for document in to_index:
            started = time.perf_counter()
            result = indexer.index_document(
                document, session_id=SESSION_ID if document != private else "speed-private"
            )
            indexing.append(
                {
                    "document": "private sample.pdf" if document == private else document.name,
                    "kilobytes": round(document.stat().st_size / 1024, 1),
                    "chunks": result.chunk_count,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )

        started = time.perf_counter()
        reranker = CrossEncoderReranker(
            base_retriever=build_session_retriever(
                settings=settings, embedding_model=embedding, vector_store=store
            ),
            model_name=settings.reranker_model,
            candidate_pool_size=settings.reranker_candidate_pool_size,
            device=settings.reranker_device,
        )
        reranker.model.predict([("warm", "up")])
        reranker_load_s = time.perf_counter() - started

        watch = Stopwatch()
        reranker.base_retriever = TimedRetriever(reranker.base_retriever, watch, "search_ms")
        llm = TimedLLM(build_llm(settings), watch)
        if not llm.health_check():
            raise SystemExit(f"{provider} ({settings.llm_model}) is not reachable.")
        pipeline = RAGPipeline(
            retriever=TimedRetriever(reranker, watch, "retrieval_ms"),
            llm=llm,
            confidence_gate=RetrievalConfidenceGate(min_top_score=settings.answerability_min_score),
            grounding_verifier=TimedVerifier(
                GroundingVerifier(
                    scorer=reranker.model, min_score=settings.active_grounding_min_score
                ),
                watch,
            ),
            injection_filter=settings.injection_filter_enabled,
        )

        interval = GEMINI_MIN_INTERVAL_SECONDS if provider == "gemini" else 0.0
        filters: dict[str, str | int] = {"session_id": SESSION_ID}
        answers = []
        for number, case in enumerate(cases, start=1):
            # A quota error restarts the question, so waiting never counts as answer time.
            for wait in (*RETRY_WAITS_SECONDS, None):
                watch.ms = {}
                llm.calls, llm.usage = 0, []
                started = time.perf_counter()
                first_token_ms = None
                final: RAGResponse | None = None
                try:
                    for item in pipeline.stream_with_citations(
                        case.question, top_k=TOP_K, metadata_filters=filters
                    ):
                        if isinstance(item, RAGResponse):
                            final = item
                        elif first_token_ms is None:
                            first_token_ms = (time.perf_counter() - started) * 1000.0
                    break
                except (LLMRateLimitError, LLMConnectionError) as exc:
                    if wait is None:
                        raise
                    print(f"  {type(exc).__name__}; retrying in {wait}s", flush=True)
                    time.sleep(wait)
            total_ms = (time.perf_counter() - started) * 1000.0
            assert final is not None
            ms = watch.ms
            row = {
                "case_id": case.case_id,
                "llm_calls": llm.calls,
                "answered": final.answer != ABSTENTION_ANSWER,
                "search_ms": round(ms.get("search_ms", 0.0), 1),
                "rerank_ms": round(ms.get("retrieval_ms", 0.0) - ms.get("search_ms", 0.0), 1),
                "first_token_ms": None if first_token_ms is None else round(first_token_ms, 1),
                "generation_ms": round(ms.get("generation_ms", 0.0), 1),
                "grounding_ms": round(ms.get("grounding_ms", 0.0), 1),
                "total_ms": round(total_ms, 1),
                "input_tokens": sum(u.input_tokens for u in llm.usage),
                "output_tokens": sum(u.output_tokens for u in llm.usage),
                "thinking_tokens": sum(u.thinking_tokens for u in llm.usage),
            }
            answers.append(row)
            print(
                f"[{provider} {number}/{len(cases)}] {case.case_id} total={row['total_ms']:.0f}ms "
                f"first_token={row['first_token_ms']} tokens={row['input_tokens']}"
                f"+{row['output_tokens']}",
                flush=True,
            )
            if llm.calls:
                time.sleep(max(0.0, interval - total_ms / 1000.0))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    measured = {
        "provider": provider,
        "model": settings.llm_model,
        "hardware": hardware(),
        "startup_seconds": {
            "embedding_model": round(embedding_load_s, 2),
            "reranker_model": round(reranker_load_s, 2),
        },
        "indexing": indexing,
        "answers": answers,
    }
    out = out_dir / "run.json"
    out.write_text(json.dumps(measured, indent=2) + "\n", encoding="utf-8")
    return out


def percentile(values: list[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def summarize(run_result: dict[str, Any]) -> dict[str, Any]:
    answers = run_result["answers"]
    with_llm = [a for a in answers if a["llm_calls"]]
    stages = {}
    for stage in STAGES:
        rows = with_llm if stage in ("first_token_ms", "generation_ms") else answers
        values = [a[stage] for a in rows if a[stage] is not None and a[stage] > 0]
        stages[stage] = {
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "count": len(values),
        }
    tokens = {
        key: statistics.mean(a[key] for a in with_llm) if with_llm else 0.0
        for key in ("input_tokens", "output_tokens", "thinking_tokens")
    }
    per_question = cost(tokens, run_result["model"])
    price = PRICES.get(run_result["model"])
    return {
        "provider": run_result["provider"],
        "model": run_result["model"],
        "hardware": run_result["hardware"],
        "startup_seconds": run_result["startup_seconds"],
        "indexing": run_result["indexing"],
        "questions": len(answers),
        "questions_reaching_llm": len(with_llm),
        "llm_calls_per_question": statistics.mean(a["llm_calls"] for a in with_llm)
        if with_llm
        else 0.0,
        "stages_ms": stages,
        "mean_tokens_per_question": {k: round(v, 1) for k, v in tokens.items()},
        "cost_usd": None
        if per_question is None
        else {
            "per_question": round(per_question, 6),
            "per_1000_questions": round(per_question * 1000, 2),
            "prices": price,
        },
    }


def analyze() -> dict[str, Any]:
    summaries = {}
    for path in sorted(RESULTS_DIR.glob("*/run.json")):
        summaries[path.parent.name] = summarize(json.loads(path.read_text("utf-8")))
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    for name, s in summaries.items():
        print(f"\n{name} ({s['model']}) on {s['hardware']['processor']}")
        print(f"  startup: {s['startup_seconds']}")
        for row in s["indexing"]:
            print(f"  index {row['document']}: {row['seconds']} s, {row['chunks']} chunks")
        for stage, v in s["stages_ms"].items():
            print(f"  {stage:<16} median {v['median']} ms   p95 {v['p95']} ms   (n={v['count']})")
        print(f"  tokens per question: {s['mean_tokens_per_question']}")
        print(f"  cost: {s['cost_usd']}")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--provider", choices=PROVIDERS, default="gemini")
    run_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run_parser.add_argument("--include-private", action="store_true")
    sub.add_parser("analyze")
    args = parser.parse_args()
    if args.command == "run":
        print(f"wrote {run(args.provider, args.ollama_url, args.include_private)}")
    else:
        analyze()


if __name__ == "__main__":
    main()
