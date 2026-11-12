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

## 12. Public benchmark and the answerability gap

`data/evaluation/public/` holds a public benchmark that CI can run: four fictional company documents (HR handbook, IT security policy, product warranty and cloud SLA) and 75 cases. 59 answerable questions are paraphrased so that they do not share wording with the evidence, and 16 unanswerable questions are plausible but not covered by any document. A retrieved chunk counts as relevant if it comes from the labelled document and contains one of the labelled verbatim evidence phrases. A unit test checks that every phrase really exists.

`python -m docqna.evaluation.public_benchmark` indexes the corpus with the production retrieval stack (fixed chunking, MiniLM, dense + per-session BM25, RRF, cross-encoder) and scores retrieval and the confidence gate, without an LLM:

| Metric | Value |
|---|---:|
| Hit@1 / Hit@3 / Hit@5 | 0.746 / 0.983 / 1.000 |
| MRR | 0.863 |
| Answerable questions passing the gate | 0.424 |
| Unanswerable questions correctly abstained | 0.875 |

**Finding.** Retrieval generalises to a new domain and to paraphrased questions. The answerability threshold does not. The cross-encoder's rank-1 scores for answerable paraphrases range from about -10 to +8. Unanswerable questions mostly score below -8, but some reach +3.5, so no single threshold separates the two well: at `0`, 58% of answerable questions are refused. The original threshold was tuned on the 30-case private set, whose questions were closer to the document's wording.

**Next step (not yet applied).** Recalibrate answerability end to end with the LLM in the loop: lower the retrieval gate and rely on the model's own abstention plus citation validation. Measure both answer correctness and false answers on unanswerable questions before changing the production default.

CI runs the benchmark and fails if any metric drops below `data/evaluation/public/thresholds.json`. The thresholds are regression floors set just below today's values.
