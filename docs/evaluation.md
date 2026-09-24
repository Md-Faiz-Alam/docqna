# DocQ&A Evaluation

## Evaluation philosophy

DocQ&A evaluates retrieval, answerability, and final RAG quality separately so failures can be attributed to the correct layer.

## Dataset

The final evaluation set contains 30 labeled cases:

- 18 answerable
- 12 unanswerable

Versioned datasets:

```text
data/evaluation/baseline_questions.json
data/evaluation/evaluation_dataset_v2.json
data/evaluation/unanswerable_questions.json
```

## Retrieval results

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR | Page Recall |
|---|---:|---:|---:|---:|---:|
| Dense | 0.6667 | 0.8889 | 1.0000 | 0.7963 | 0.9444 |
| BM25 | 0.7778 | 1.0000 | 1.0000 | 0.8889 | 0.9444 |
| RRF | 0.7222 | 1.0000 | 1.0000 | 0.8519 | 0.9444 |
| RRF + Cross-Encoder | 0.8333 | 1.0000 | 1.0000 | 0.9074 | 1.0000 |

BM25 measured nDCG at approximately `0.9067`, fractionally above the RRF + cross-encoder result of approximately `0.9050`. The project therefore does not claim that reranking improves every metric.

## Chunking

Compared fixed, structure-aware, and semantic strategies.

The fixed strategy was retained for the production path as the best measured quality/complexity balance for this dataset.

```text
chunk size: 200 words
overlap: 40 words
```

## Embeddings

Compared:

- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`

Measured retrieval quality was effectively identical on this dataset. `all-MiniLM-L6-v2` was retained because it is smaller and cheaper while preserving measured quality here.

## Top-k

The selected production value is:

```text
top_k = 5
```

Smaller context was cleaner but could miss evidence; larger context did not improve recall on the evaluated cases.

## Answerability calibration

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

This is dataset-specific evidence, not a universal guarantee.

## Final artifacts

```text
data/evaluation/results/phase22g_final_analysis/
```

contains the final experiment matrix, error analysis, manifest, report, and charts.

## Supported conclusions

The current evidence supports:

- fixed chunking for the default production path,
- `all-MiniLM-L6-v2`,
- dense + BM25 retrieval,
- RRF fusion,
- cross-encoder reranking,
- top-5 context,
- CE threshold `0` for the current answerability gate.

## Not proven

The current results do not prove:

- universal superiority across domains,
- universal correctness of threshold `0`,
- reranking superiority on every metric,
- robustness to scanned/OCR-heavy corpora,
- large-scale multi-tenant performance.
