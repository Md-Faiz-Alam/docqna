# DocQ&A Evaluation

---

## 1. Evaluation philosophy

DocQ&A evaluates retrieval, answerability, and final RAG quality separately so failures can be attributed to the correct layer.

The evaluation results below support the frozen production RAG configuration. Operational Docker validation is tracked separately because packaging and persistence tests do not change retrieval-quality metrics.

---

## 2. Dataset

The final evaluation set contains 30 labeled cases:

- 18 answerable
- 12 unanswerable

Versioned datasets:

```text
data/evaluation/baseline_questions.json
data/evaluation/evaluation_dataset_v2.json
data/evaluation/unanswerable_questions.json
```

---

## 3. Retrieval results

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Page Recall |
|---|---:|---:|---:|---:|---:|
| Dense | 0.6667 | 0.8889 | 1.0000 | 0.7963 | 0.9444 |
| BM25 | 0.7778 | 1.0000 | 1.0000 | 0.8889 | 0.9444 |
| RRF | 0.7222 | 1.0000 | 1.0000 | 0.8519 | 0.9444 |
| RRF + Cross-Encoder | 0.8333 | 1.0000 | 1.0000 | 0.9074 | 1.0000 |

The RRF + cross-encoder configuration produced the strongest measured Hit@1, MRR, and Page Recall among the compared retrieval configurations.

BM25 measured nDCG at approximately `0.9067`, fractionally above the RRF + cross-encoder result of approximately `0.9050`. The project therefore does not claim that reranking improves every metric.

---

## 4. Chunking

Compared:

- fixed chunking
- structure-aware chunking
- semantic chunking

The fixed strategy was retained for the production path as the best measured quality/complexity balance for this dataset.

```text
chunk size: 200 words
overlap: 40 words
```

---

## 5. Embeddings

Compared:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`

Measured retrieval quality was effectively identical on this dataset. `all-MiniLM-L6-v2` was retained because it is smaller and cheaper while preserving the measured quality here.

---

## 6. Final context size

The selected production value is:

```text
top_k = 5
```

A smaller context was cleaner but could miss evidence. Larger contexts did not improve recall on the evaluated cases.

---

## 7. Answerability calibration

Observed on the 30-case evaluation set:

```text
minimum answerable score   ≈  1.335
maximum unanswerable score ≈ -7.574
```

Production threshold:

```text
0
```

This achieved perfect observed answerable/unanswerable classification on the current dataset.

The result is dataset-specific evidence, not a universal guarantee. The threshold should be recalibrated if the domain, reranker, chunking, or retrieval behavior changes materially.

---

## 8. Final artifacts

```text
data/evaluation/results/phase22g_final_analysis/
```

This directory contains the final experiment matrix, error analysis, manifest, report, and charts used to support the production configuration.

---

## 9. Supported conclusions

The current evidence supports:

- fixed chunking with 200-word chunks and 40-word overlap
- `sentence-transformers/all-MiniLM-L6-v2`
- dense + BM25 retrieval
- RRF fusion with `k=60`
- cross-encoder reranking
- top-5 final context
- rank-1 CE threshold `0` for the current answerability gate

---

## 10. Not proven

The current results do not prove:

- universal superiority across domains
- universal correctness of threshold `0`
- reranking superiority on every metric
- robustness to scanned/OCR-heavy corpora
- large-scale multi-tenant performance
- public-cloud reliability or internet-facing security

---

## 11. Deployment validation note

The Docker deployment work did not change the frozen production RAG configuration or the evaluation results above.

The containerized stack was validated separately through functional end-to-end checks covering:

- document upload and indexing
- grounded answer generation with citations
- exact abstention for an unrelated question
- browser-session continuity across page refresh
- persistent Chroma and document-registry storage
- QA after API-container recreation without re-upload

These are operational validation checks, not additional benchmark cases. They do not extend or replace the 30-case evaluation dataset.
