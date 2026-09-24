from docqna.constants import (
    CHROMA_INDEX_DIR,
    DEFAULT_CHROMA_COLLECTION,
)
from docqna.embeddings import SentenceTransformerEmbedding
from docqna.generation.providers.base import BaseLLMProvider
from docqna.indexing import ChromaVectorStore
from docqna.rag import RAGPipeline
from docqna.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    RetrievalConfidenceGate,
    RRFHybridRetriever,
)
from docqna.settings import get_settings


ABSTENTION_ANSWER = (
    "The provided documents do not contain "
    "enough information to answer the question."
)


class CountingTestLLM(BaseLLMProvider):
    """
    Deterministic test LLM.

    This lets us verify that abstained questions
    never reach generation.
    """

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        self.generate_calls += 1

        return (
            "The retrieved document contains relevant "
            "information for this question [1]."
        )

    def health_check(self) -> bool:
        return True


settings = get_settings()


# --------------------------------
# 1. Embedding model
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    model_name=settings.embedding_model,
    device=settings.embedding_device,
    batch_size=settings.embedding_batch_size,
)


# --------------------------------
# 2. Chroma corpus
# --------------------------------

vector_store = ChromaVectorStore(
    persist_directory=CHROMA_INDEX_DIR,
    collection_name=DEFAULT_CHROMA_COLLECTION,
)


chunks = vector_store.get_all_chunks()


print(
    "Chroma records:",
    vector_store.count(),
)


assert len(chunks) == 192


# --------------------------------
# 3. Dense retrieval
# --------------------------------

dense_retriever = DenseRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store,
)


# --------------------------------
# 4. BM25 retrieval
# --------------------------------

bm25_retriever = BM25Retriever(
    chunks=chunks,
)


# --------------------------------
# 5. RRF
# --------------------------------

hybrid_retriever = RRFHybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=bm25_retriever,
    rrf_k=60,
    candidate_pool_size=10,
)


# --------------------------------
# 6. Cross-encoder reranker
# --------------------------------

reranker = CrossEncoderReranker(
    base_retriever=hybrid_retriever,
    model_name=(
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    ),
    candidate_pool_size=20,
    device="cpu",
)


# --------------------------------
# 7. Confidence gate
# --------------------------------

confidence_gate = RetrievalConfidenceGate(
    min_top_score=0.0
)


# --------------------------------
# 8. Test LLM
# --------------------------------

llm = CountingTestLLM()


# --------------------------------
# 9. RAG pipeline
# --------------------------------

rag = RAGPipeline(
    retriever=reranker,
    llm=llm,
    confidence_gate=confidence_gate,
)


# --------------------------------
# 10. Answerable question
# --------------------------------

answerable_question = (
    "What are the four pillars of the "
    "PMI Code of Ethics and Professional Conduct?"
)


print(
    "\nAnswerable question:"
)

print(
    answerable_question
)


response = rag.ask_with_citations(
    query=answerable_question,
    top_k=5,
)


print(
    "Answer:",
    response.answer,
)

print(
    "Cited IDs:",
    response.cited_source_ids,
)


assert response.answer != ABSTENTION_ANSWER

assert response.sources

assert response.cited_source_ids == [1]

assert llm.generate_calls == 1


print(
    "Answerable question accepted."
)


# --------------------------------
# 11. Unanswerable questions
# --------------------------------

unanswerable_questions = [
    "What is the capital city of Japan?",
    "Who wrote Pride and Prejudice?",
    "How does DNA replication work?",
    (
        "What is the current market price "
        "of Microsoft stock?"
    ),
    (
        "What is the latest version number "
        "of Microsoft Project?"
    ),
]


for index, question in enumerate(
    unanswerable_questions,
    start=1,
):
    print(
        f"\nUnanswerable {index}:"
    )

    print(
        question
    )

    calls_before = (
        llm.generate_calls
    )

    response = rag.ask_with_citations(
        query=question,
        top_k=5,
    )

    print(
        "Answer:",
        response.answer,
    )

    assert (
        response.answer
        == ABSTENTION_ANSWER
    )

    assert response.sources == []

    assert response.cited_source_ids == []

    # Critical check:
    # abstention must happen BEFORE LLM generation.
    assert (
        llm.generate_calls
        == calls_before
    )

    print(
        "Correctly abstained."
    )


# --------------------------------
# 12. Final validation
# --------------------------------

assert llm.generate_calls == 1


print(
    "\nLLM generation calls:",
    llm.generate_calls,
)

print(
    "Expected generation calls: 1"
)

print(
    "\nEnd-to-end abstention test passed"
)