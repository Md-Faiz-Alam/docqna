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
| Answerable questions passing the gate (at the original `0`) | 0.424 |
| Unanswerable questions correctly abstained (at the original `0`) | 0.875 |

**Finding.** Retrieval generalises to a new domain and to paraphrased questions. The answerability threshold does not. The cross-encoder's rank-1 scores for answerable paraphrases range from about -10 to +8. Unanswerable questions mostly score below -8, but some reach +3.5, so no single threshold separates the two well: at `0`, 58% of answerable questions are refused. The original threshold was tuned on the 30-case private set, whose questions were closer to the document's wording.

This was fixed in phase 23 (next section). CI runs the benchmark and fails if any metric drops below `data/evaluation/public/thresholds.json`. These thresholds are regression floors set just below the values under the current gate (`-10.5`): Hit@1 0.746, Hit@5 1.0, MRR 0.863, answerable gate pass 1.0, unanswerable gate abstain 0.312. The gate-only abstain rate is now low by design, because unanswerable questions are mostly stopped by the grounding check after generation. That check needs the LLM, so it is measured by phase 23 rather than in CI.

## 13. Answerability recalibration with the LLM in the loop

`experiments/run_phase23_answerability_recalibration.py` answers all 105 questions once with the retrieval gate **disabled** (75 public, 30 private; `phi3:mini`, temperature 0). For each question it records the rank-1 cross-encoder score, whether the model answered, and whether the answer is correct. An answer counts as correct if a cited source contains the labelled evidence, or if the answer states the labelled fact; the private set requires at least half of the expected answer terms. Any gate threshold can then be simulated offline.

**The LLM is not a safe second line of defence on its own.** With no gate, `phi3:mini` answered 10 of 16 unanswerable public questions and attached citations to unrelated passages. Examples: "France won the 2018 World Cup [1]", "the dress code is business casual [1]", "the engineering team uses Python and Java [1]". A retrieval gate alone cannot separate the cases either. Lowering it from `0` to `-4` raises correct public answers from 37% to 64%, but false answers go from 6% to 31%.

**Grounding check.** `ground` scores each generated answer, with citation markers removed, against the sources it cites, using the reranker's cross-encoder. Correct answers that restate their source score high; invented answers citing unrelated passages score very low.

The first version scored the answer against each **whole chunk**, which penalised correct answers. For example, "60 euros per day, excluding alcohol [2]" scored -3.05 against its 200-word mixed-topic chunk, but 9.22 against the sentence that states it. Grounding is therefore **sentence-level**: the best score over the cited sources' sentences and adjacent sentence pairs.

`analyze` searches (gate `T`, grounding `G`) pairs:

1. constraints: false answers on unanswerable questions no higher than the old configuration on either set, and private correct answers at least 0.94;
2. then fewest false answers, then most public correct answers, then most private correct answers;
3. prefer the higher `T` (fewer LLM calls), and take `G` at the middle of the tied range (widest margin).

An earlier version of this rule maximised public correct answers first. It selected `G = 2.2`: 71.2% correct with 1/16 false answers on the public set. Because a configuration with **no** false answers costs only one correct answer, the rule was changed to put trust first. This choice was made after seeing the data and is reported here for transparency.

The rule selects `T = -10.5`, `G = 3.3` (tied range 2.5–4.0, well clear of the highest-scoring invented answer at 2.45):

| Configuration | Public correct | Public correctly cited | Public wrong answers | Public false answers | Private correct | Private false answers |
|---|---:|---:|---:|---:|---:|---:|
| Old: gate `>= 0` | 0.373 | 0.356 | 0.034 | 0.062 | 1.000 | 0.000 |
| Gate `>= -4`, no grounding | 0.644 | 0.542 | 0.051 | 0.312 | 1.000 | 0.000 |
| Gate `>= -9`, chunk-level grounding `>= 0.7` | 0.593 | 0.593 | 0.034 | 0.062 | 1.000 | 0.000 |
| **New: gate `>= -10.5`, sentence-level grounding `>= 3.3`** | **0.695** | **0.678** | **0.017** | **0.000** | **0.944** | **0.000** |

Keeping the private set at 100% would require `G <= -2.1`. That is because of one correct but long, multi-claim answer (`quality_001`). Such a low threshold readmits invented answers and drops public correct answers back to about 36%.

**Caveats.** The thresholds were selected and reported on the same 105 questions, so the numbers are optimistic. Only 28 unanswerable questions exist across both sets, so "0 false answers" means none observed in 28. The recorded answers come from one model (`phi3:mini`); a different LLM needs a new run. All raw per-case results and the sweep are in `data/evaluation/results/phase23_recalibration/`.
