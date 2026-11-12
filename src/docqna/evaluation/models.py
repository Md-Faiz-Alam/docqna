from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EvaluationCase(BaseModel):
    """
    One manually curated evaluation question.

    Phase 21 supports both answerable and unanswerable cases.

    Backward compatibility:
    - legacy answerable cases that omit ``should_answer`` are
      inferred as answerable when they contain retrieval/answer
      labels.
    - legacy unanswerable cases that contain only ``case_id`` and
      ``question`` are inferred as unanswerable.
    """

    case_id: str
    question: str

    should_answer: bool = True

    relevant_pages: list[int] = Field(default_factory=list)

    relevant_chunk_ids: list[str] = Field(default_factory=list)

    expected_answer_terms: list[str] = Field(default_factory=list)

    reference_answer: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_answerability(cls, data: Any) -> Any:
        """
        Infer ``should_answer`` only for legacy records that do not
        explicitly provide it.
        """

        if not isinstance(data, dict):
            return data

        if "should_answer" in data:
            return data

        upgraded = dict(data)

        has_answer_labels = bool(
            upgraded.get("relevant_pages")
            or upgraded.get("relevant_chunk_ids")
            or upgraded.get("expected_answer_terms")
            or upgraded.get("reference_answer")
        )

        upgraded["should_answer"] = has_answer_labels

        return upgraded

    @model_validator(mode="after")
    def _validate_labels(self) -> "EvaluationCase":
        """Validate answerability and ground-truth consistency."""

        if not self.case_id.strip():
            raise ValueError("case_id cannot be empty.")

        if not self.question.strip():
            raise ValueError("question cannot be empty.")

        if any(page <= 0 for page in self.relevant_pages):
            raise ValueError("relevant_pages must contain positive page numbers.")

        if len(self.relevant_pages) != len(set(self.relevant_pages)):
            raise ValueError("relevant_pages cannot contain duplicates.")

        normalized_chunk_ids = [chunk_id.strip() for chunk_id in self.relevant_chunk_ids]

        if any(not chunk_id for chunk_id in normalized_chunk_ids):
            raise ValueError("relevant_chunk_ids cannot contain empty values.")

        if len(normalized_chunk_ids) != len(set(normalized_chunk_ids)):
            raise ValueError("relevant_chunk_ids cannot contain duplicates.")

        if self.should_answer:
            if not self.relevant_pages and not self.relevant_chunk_ids:
                raise ValueError(
                    "Answerable evaluation cases must define relevant_pages or relevant_chunk_ids."
                )

        else:
            if self.relevant_pages or self.relevant_chunk_ids:
                raise ValueError(
                    "Unanswerable evaluation cases cannot define "
                    "relevant_pages or relevant_chunk_ids."
                )

        return self


class RetrievalCaseResult(BaseModel):
    """Metrics and diagnostics for one retrieval case."""

    case_id: str
    question: str

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float

    reciprocal_rank: float
    page_recall_at_5: float

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    ndcg_at_k: float = 0.0

    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    retrieved_pages: list[str] = Field(default_factory=list)

    distances: list[float] = Field(default_factory=list)


class RetrievalEvaluationReport(BaseModel):
    """Aggregate dense-retrieval evaluation report."""

    num_cases: int
    top_k: int
    num_skipped_unanswerable: int = 0

    mean_hit_at_1: float
    mean_hit_at_3: float
    mean_hit_at_5: float

    mean_reciprocal_rank: float
    mean_page_recall_at_5: float

    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mean_ndcg_at_k: float = 0.0

    cases: list[RetrievalCaseResult] = Field(default_factory=list)


class AnswerCaseResult(BaseModel):
    """Evaluation result for one generated RAG answer."""

    case_id: str
    question: str

    answer: str

    expected_answer_terms: list[str] = Field(default_factory=list)

    matched_answer_terms: list[str] = Field(default_factory=list)

    answer_term_coverage: float

    citation_presence: float
    citation_validity: float

    cited_source_ids: list[int] = Field(default_factory=list)

    matched_cited_evidence_terms: list[str] = Field(default_factory=list)

    cited_evidence_term_coverage: float

    answer_word_count: int

    elapsed_seconds: float = Field(ge=0.0)

    error: str | None = None


class AnswerEvaluationReport(BaseModel):
    """Aggregate answer and citation evaluation report."""

    num_cases: int
    successful_cases: int
    failed_cases: int
    top_k: int

    mean_answer_term_coverage: float
    mean_citation_presence: float
    mean_citation_validity: float
    mean_cited_evidence_term_coverage: float

    mean_answer_word_count: float

    mean_case_elapsed_seconds: float
    total_case_elapsed_seconds: float

    cases: list[AnswerCaseResult] = Field(default_factory=list)


class AnswerabilityCaseResult(BaseModel):
    """Answerability/abstention outcome for one evaluation case."""

    case_id: str
    question: str

    should_answer: bool
    predicted_should_answer: bool | None = None
    abstained: bool | None = None
    correct: bool = False

    elapsed_seconds: float = Field(ge=0.0)

    error: str | None = None


class AnswerabilityEvaluationReport(BaseModel):
    """Aggregate answerability and abstention evaluation."""

    num_cases: int
    successful_cases: int
    error_cases: int
    top_k: int

    true_answers: int
    true_abstentions: int
    false_answers: int
    false_abstentions: int

    accuracy: float

    answer_precision: float
    answer_recall: float
    answer_f1: float

    abstention_precision: float
    abstention_recall: float
    abstention_f1: float

    mean_case_elapsed_seconds: float
    total_case_elapsed_seconds: float

    cases: list[AnswerabilityCaseResult] = Field(default_factory=list)


class RAGQualityScores(BaseModel):
    """LLM-as-judge quality scores for one answered case."""

    answer_relevance: float = Field(ge=0.0, le=1.0)

    context_relevance: float = Field(ge=0.0, le=1.0)

    faithfulness: float = Field(ge=0.0, le=1.0)

    citation_correctness: float = Field(ge=0.0, le=1.0)

    rationale: str = ""


class RAGQualityCaseResult(BaseModel):
    """RAG-quality outcome for one evaluation case."""

    case_id: str
    question: str

    should_answer: bool

    status: Literal["judged", "abstained", "skipped_unanswerable", "error"]

    answer: str = ""

    answer_relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    context_relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)

    citation_correctness: float | None = Field(default=None, ge=0.0, le=1.0)

    rationale: str = ""

    cited_source_ids: list[int] = Field(default_factory=list)

    source_count: int = Field(default=0, ge=0)

    generation_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    judge_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    total_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    error: str | None = None


class RAGQualityEvaluationReport(BaseModel):
    """Aggregate RAG-quality evaluation report."""

    num_cases: int
    answerable_cases: int
    skipped_unanswerable_cases: int

    judged_cases: int
    abstained_cases: int
    failed_cases: int

    top_k: int

    mean_answer_relevance: float
    mean_context_relevance: float
    mean_faithfulness: float
    mean_citation_correctness: float

    mean_generation_elapsed_seconds: float
    mean_judge_elapsed_seconds: float
    mean_total_elapsed_seconds: float
    total_elapsed_seconds: float

    cases: list[RAGQualityCaseResult] = Field(default_factory=list)


class SystemEvaluationCaseResult(BaseModel):
    """System-level latency and outcome metrics for one RAG case."""

    case_id: str
    question: str
    should_answer: bool

    success: bool
    abstained: bool | None = None

    retrieval_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    generation_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    postprocessing_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    total_elapsed_seconds: float = Field(default=0.0, ge=0.0)

    source_count: int = Field(default=0, ge=0)

    cited_source_count: int = Field(default=0, ge=0)

    error: str | None = None


class SystemEvaluationReport(BaseModel):
    """Aggregate system-level RAG evaluation metrics."""

    num_cases: int
    successful_cases: int
    failed_cases: int

    answered_cases: int
    abstained_cases: int

    top_k: int

    success_rate: float = Field(ge=0.0, le=1.0)

    error_rate: float = Field(ge=0.0, le=1.0)

    abstention_rate: float = Field(ge=0.0, le=1.0)

    mean_retrieval_elapsed_seconds: float = Field(ge=0.0)

    mean_generation_elapsed_seconds: float = Field(ge=0.0)

    mean_postprocessing_elapsed_seconds: float = Field(ge=0.0)

    mean_total_elapsed_seconds: float = Field(ge=0.0)

    total_elapsed_seconds: float = Field(ge=0.0)

    cases: list[SystemEvaluationCaseResult] = Field(default_factory=list)
