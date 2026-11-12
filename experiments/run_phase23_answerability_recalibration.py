"""Phase 23: recalibrate the answerability gate end to end, with the LLM in the loop.

Phase 22f chose the rank-1 cross-encoder threshold (>= 0) from retrieval
scores alone, on one private document. The public benchmark showed it refuses
most paraphrased answerable questions. Here every case is answered once with
the gate *disabled*, recording:

- the rank-1 cross-encoder score (what the gate looks at), and
- what the full pipeline does without the gate: abstain, or answer, and if so
  whether a cited source contains the labelled evidence.

Because the gate only decides "skip the LLM or not", the outcome for any
threshold T can then be simulated offline: score < T means abstain, otherwise
use the recorded LLM outcome.

The LLM alone proved a poor second line of defence (phi3:mini answered 10/16
unanswerable public questions, citing unrelated sources), so `ground` also
scores each recorded answer against the sources it cites with the reranker's
cross-encoder. `analyze` then searches (retrieval gate T, grounding threshold G)
pairs under a fixed rule: no more false answers than the old configuration on
either dataset, private recall >= 0.94; maximise public correct answers; among
ties prefer the higher T, then the G with the widest margin.

    python -m experiments.run_phase23_answerability_recalibration run --dataset public
    python -m experiments.run_phase23_answerability_recalibration run --dataset private
    python -m experiments.run_phase23_answerability_recalibration ground --dataset public
    python -m experiments.run_phase23_answerability_recalibration ground --dataset private
    python -m experiments.run_phase23_answerability_recalibration analyze
"""

import argparse
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from docqna.api.dependencies import build_llm, build_session_retriever
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.evaluation.public_benchmark import is_relevant as public_is_relevant
from docqna.evaluation.public_benchmark import load_cases as load_public_cases
from docqna.evaluation.runner import load_evaluation_cases
from docqna.indexing import ChromaVectorStore, DocumentIndexService
from docqna.ingestion import DocumentIngestionService, DocumentRegistry
from docqna.rag import ABSTENTION_ANSWER, RAGPipeline
from docqna.rag.grounding import GroundingVerifier
from docqna.rag.models import CitationSource
from docqna.retrieval import CrossEncoderReranker
from docqna.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results" / "phase23_recalibration"
SESSION_ID = "phase23-recalibration-" + "0" * 32
TOP_K = 5


@dataclass
class Case:
    case_id: str
    question: str
    should_answer: bool
    is_relevant: Callable[[CitationSource], bool]
    key_facts: list[str]


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def public_cases() -> tuple[list[Path], list[Case]]:
    corpus = sorted((PROJECT_ROOT / "data/evaluation/public/corpus").glob("*.txt"))
    cases = []
    for case in load_public_cases():

        def relevant(source: CitationSource, case=case) -> bool:
            return public_is_relevant(source, case)  # type: ignore[arg-type]

        facts = [n for phrase in case.evidence for n in _numbers(phrase)]
        cases.append(Case(case.case_id, case.question, case.should_answer, relevant, facts))
    return corpus, cases


def private_cases() -> tuple[list[Path], list[Case]]:
    sample = PROJECT_ROOT / "data/samples/sample.pdf"
    if not sample.exists():
        raise SystemExit("data/samples/sample.pdf is required for the private dataset.")
    cases = []
    for case in load_evaluation_cases(PROJECT_ROOT / "data/evaluation/evaluation_dataset_v2.json"):
        pages = set(case.relevant_pages)

        def relevant(source: CitationSource, pages=pages) -> bool:
            if source.page_start is None:
                return False
            end = source.page_end or source.page_start
            return any(source.page_start <= page <= end for page in pages)

        cases.append(Case(case.case_id, case.question, case.should_answer, relevant, []))
    return [sample], cases


def _build_retriever(
    documents: list[Path], settings: Settings, work_dir: Path
) -> CrossEncoderReranker:
    """Index `documents` with the production stack and return the reranker."""

    embedding = SentenceTransformerEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    store = ChromaVectorStore(persist_directory=work_dir / "chroma", collection_name="phase23")
    indexer = DocumentIndexService(
        ingestion_service=DocumentIngestionService(),
        registry=DocumentRegistry(work_dir / "registry.db"),
        embedding_model=embedding,
        vector_store=store,
    )
    for document in documents:
        indexer.index_document(document, session_id=SESSION_ID)

    return CrossEncoderReranker(
        base_retriever=build_session_retriever(
            settings=settings, embedding_model=embedding, vector_store=store
        ),
        model_name=settings.reranker_model,
        candidate_pool_size=settings.reranker_candidate_pool_size,
        device=settings.reranker_device,
    )


def run(dataset: str, ollama_url: str) -> Path:
    documents, cases = public_cases() if dataset == "public" else private_cases()
    out = RESULTS_DIR / f"{dataset}_cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.read_text("utf-8").splitlines()}

    settings = Settings(_env_file=None, ollama_base_url=ollama_url)  # type: ignore[call-arg]
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-phase23-"))
    try:
        retriever = _build_retriever(documents, settings, work_dir)
        llm = build_llm(settings)
        if not llm.health_check():
            raise SystemExit(f"Ollama is not reachable at {ollama_url}.")
        ungated = RAGPipeline(retriever=retriever, llm=llm, confidence_gate=None)
        filters = {"session_id": SESSION_ID}

        with out.open("a", encoding="utf-8") as sink:
            for number, case in enumerate(cases, start=1):
                if case.case_id in done:
                    continue
                top = retriever.retrieve(case.question, top_k=TOP_K, metadata_filters=filters)
                top_score = top[0].score if top else None
                response = ungated.ask_with_citations(
                    case.question, top_k=TOP_K, metadata_filters=filters
                )
                answered = response.answer != ABSTENTION_ANSWER
                cited = [s for s in response.sources if s.citation_id in response.cited_source_ids]
                record = {
                    "case_id": case.case_id,
                    "should_answer": case.should_answer,
                    "top_score": top_score,
                    "answered": answered,
                    "cited_relevant": answered and any(case.is_relevant(s) for s in cited),
                    "has_key_facts": (
                        all(fact in response.answer for fact in case.key_facts)
                        if answered and case.key_facts
                        else None
                    ),
                    "answer": response.answer,
                }
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                print(
                    f"[{dataset} {number}/{len(cases)}] {case.case_id} "
                    f"score={top_score:.2f} answered={answered} "
                    f"cited_relevant={record['cited_relevant']}",
                    flush=True,
                )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return out


def ground(dataset: str) -> Path:
    """Add `grounding_score` (cross-encoder: answer vs. cited sources) to each record."""

    documents, cases = public_cases() if dataset == "public" else private_cases()
    questions = {case.case_id: case.question for case in cases}
    path = RESULTS_DIR / f"{dataset}_cases.jsonl"
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    work_dir = Path(tempfile.mkdtemp(prefix="docqna-phase23-"))
    try:
        retriever = _build_retriever(documents, settings, work_dir)
        verifier = GroundingVerifier(scorer=retriever.model, min_score=0.0)
        filters = {"session_id": SESSION_ID}
        for record in records:
            record["grounding_score"] = None
            if not record["answered"]:
                continue
            # Retrieval is deterministic, so this reproduces the sources the
            # recorded answer's citation numbers refer to.
            results = retriever.retrieve(
                questions[record["case_id"]], top_k=TOP_K, metadata_filters=filters
            )
            cited = [int(n) for n in re.findall(r"\[(\d+)\]", record["answer"])]
            texts = [results[i - 1].text for i in cited if 1 <= i <= len(results)]
            record["grounding_score"] = verifier.evaluate(record["answer"], texts).score
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    return path


def outcomes_at(
    records: list[dict], threshold: float, grounding_min: float | None = None
) -> dict[str, float]:
    """Simulate the gate at `threshold` over recorded ungated outcomes."""

    answerable = [r for r in records if r["should_answer"]]
    unanswerable = [r for r in records if not r["should_answer"]]

    def passes(r: dict) -> bool:
        if r["top_score"] is None or r["top_score"] < threshold:
            return False
        if grounding_min is not None and r["answered"]:
            score = r.get("grounding_score")
            return score is not None and score >= grounding_min
        return True

    def is_correct(r: dict) -> bool:
        # `correct` (answer text states the labelled fact, or a cited source holds
        # the evidence) is added by analyze(); fall back to citation evidence only.
        return bool(r.get("correct", r["cited_relevant"]))

    correct = sum(passes(r) and is_correct(r) for r in answerable)
    correctly_cited = sum(passes(r) and r["cited_relevant"] for r in answerable)
    wrong = sum(passes(r) and r["answered"] and not is_correct(r) for r in answerable)
    false_answers = sum(passes(r) and r["answered"] for r in unanswerable)

    def rate(count: int, group: list[dict]) -> float:
        return count / len(group) if group else 0.0

    return {
        "threshold": threshold,
        "grounding_min": grounding_min,
        "answerable_correct": rate(correct, answerable),
        "answerable_correctly_cited": rate(correctly_cited, answerable),
        "answerable_wrong_answer": rate(wrong, answerable),
        "unanswerable_false_answer": rate(false_answers, unanswerable),
    }


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def answer_checkers() -> dict[str, dict[str, Callable[[str], bool]]]:
    """Per dataset and case: does the answer text state the labelled fact?"""

    public: dict[str, Callable[[str], bool]] = {}
    for case in load_public_cases():
        if case.should_answer:
            phrases = [_norm(p) for p in case.evidence]
            facts = [n for p in case.evidence for n in _numbers(p)]

            def check(answer: str, phrases=phrases, facts=facts) -> bool:
                text = _norm(answer)
                return any(p in text for p in phrases) or bool(
                    facts and all(f in text for f in facts)
                )

            public[case.case_id] = check

    private: dict[str, Callable[[str], bool]] = {}
    dataset = PROJECT_ROOT / "data/evaluation/evaluation_dataset_v2.json"
    for case in load_evaluation_cases(dataset):
        terms = [_norm(t) for t in case.expected_answer_terms]
        if case.should_answer and terms:

            def check(answer: str, terms=terms) -> bool:
                text = _norm(answer)
                return sum(t in text for t in terms) / len(terms) >= 0.5

            private[case.case_id] = check

    return {"public": public, "private": private}


def analyze() -> None:
    thresholds = [float("-inf")] + [x / 2 for x in range(-24, 9)]
    checkers = answer_checkers()
    summary: dict[str, list[dict]] = {}
    records_by_dataset: dict[str, list[dict]] = {}
    for path in sorted(RESULTS_DIR.glob("*_cases.jsonl")):
        records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        dataset = path.stem.removesuffix("_cases")
        for r in records:
            check = checkers.get(dataset, {}).get(r["case_id"])
            r["correct"] = r["answered"] and (
                r["cited_relevant"] or bool(check and check(r["answer"]))
            )
        summary[dataset] = [outcomes_at(records, t) for t in thresholds]
        records_by_dataset[dataset] = records

        print(f"\n== {dataset} ({len(records)} cases)")
        print(f"{'threshold':>9} {'correct':>8} {'cited-ok':>9} {'wrong':>7} {'false-ans':>10}")
        for row in summary[dataset]:
            print(
                f"{row['threshold']:>9} {row['answerable_correct']:>8.3f} "
                f"{row['answerable_correctly_cited']:>9.3f} "
                f"{row['answerable_wrong_answer']:>7.3f} "
                f"{row['unanswerable_false_answer']:>10.3f}"
            )

    selection = select_configuration(records_by_dataset)
    print("\n== selected configuration")
    print(json.dumps(selection, indent=2))

    (RESULTS_DIR / "threshold_sweep.json").write_text(
        json.dumps({"gate_only": summary, "selection": selection}, indent=2, default=str),
        encoding="utf-8",
    )


def select_configuration(records_by_dataset: dict[str, list[dict]]) -> dict:
    """Pick (T, G) with the documented rule; report the old configuration too."""

    public, private = records_by_dataset["public"], records_by_dataset["private"]
    baseline_public = outcomes_at(public, 0.0)
    baseline_private = outcomes_at(private, 0.0)

    candidates = []
    for t in [x / 2 for x in range(-24, 1)]:
        for g in [x / 10 for x in range(-40, 41)]:
            pub, priv = outcomes_at(public, t, g), outcomes_at(private, t, g)
            if (
                pub["unanswerable_false_answer"] <= baseline_public["unanswerable_false_answer"]
                and priv["unanswerable_false_answer"]
                <= baseline_private["unanswerable_false_answer"]
                and priv["answerable_correct"] >= 0.94
            ):
                candidates.append((t, g, pub, priv))

    def quality(c: tuple) -> tuple[float, float, float]:
        _, _, pub, priv = c
        false_answers = pub["unanswerable_false_answer"] + priv["unanswerable_false_answer"]
        # Trust first: fewest false answers, then most correct public, then private.
        return (-false_answers, pub["answerable_correct"], priv["answerable_correct"])

    best = max(quality(c) for c in candidates)
    best_t = max(c[0] for c in candidates if quality(c) == best)
    tied = sorted(
        (c for c in candidates if quality(c) == best and c[0] == best_t), key=lambda c: c[1]
    )

    # Widest margin: the middle of the contiguous range of equally good G values.
    t, g, pub, priv = tied[len(tied) // 2]
    return {
        "rule": "constraints: false answers <= old config on both sets, private correct >= "
        "0.94; then fewest false answers, most public correct, most private correct; "
        "prefer higher T; G at the middle of the tied range",
        "old": {"threshold": 0.0, "public": baseline_public, "private": baseline_private},
        "selected": {"threshold": t, "grounding_min": g, "public": pub, "private": priv},
        "tied_grounding_range": [tied[0][1], tied[-1][1]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--dataset", choices=["public", "private"], required=True)
    run_parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ground_parser = sub.add_parser("ground")
    ground_parser.add_argument("--dataset", choices=["public", "private"], required=True)
    sub.add_parser("analyze")
    args = parser.parse_args()

    if args.command == "run":
        print(f"wrote {run(args.dataset, args.ollama_url)}")
    elif args.command == "ground":
        print(f"updated {ground(args.dataset)}")
    else:
        analyze()


if __name__ == "__main__":
    main()
