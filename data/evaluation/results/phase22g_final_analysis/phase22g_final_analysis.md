# DocQ&A Phase 22G — Final Experiment Analysis

## Final production configuration

- Chunking: fixed 200 words with 40-word overlap.
- Embedding: sentence-transformers/all-MiniLM-L6-v2 on CPU.
- Retrieval: dense + BM25 -> RRF(k=60) -> CrossEncoder reranker.
- Reranker candidate pool: 20.
- Final context top_k: 5.
- Answerability gate: rank-1 cross-encoder score with threshold 0.0.

## 1. Chunking strategy comparison

| Strategy | Chunks | Avg words | Hit@1 | MRR@5 | Precision@5 | Recall@5 | nDCG@5 | Page Recall@5 | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 192 | 199.79 | 0.8333 | 0.9074 | 0.5111 | 1.0000 | 0.9050 | 1.0000 | 0.7545 |
| structure_aware | 198 | 194.95 | 0.7222 | 0.8519 | 0.5444 | 1.0000 | 0.8786 | 1.0000 | 0.7384 |
| semantic | 641 | 87.86 | 0.7778 | 0.8889 | 0.6889 | 0.8889 | 0.8948 | 0.8889 | 0.3816 |

Fixed 200/40 retained full Recall@5 and Page Recall@5 (1.0000 / 1.0000) while also producing Hit@1=0.8333 and MRR@5=0.9074. Semantic chunking increased page-labeled precision but reduced Recall@5 to 0.8889 and Page Recall@5 to 0.8889.

## 2. Retrieval architecture comparison

| Retriever | Hit@1 | MRR@5 | Precision@5 | Recall@5 | nDCG@5 | Page Recall@5 | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0.6667 | 0.7963 | 0.5000 | 0.9444 | 0.8418 | 0.9444 | 0.0116 |
| bm25 | 0.7778 | 0.8889 | 0.4667 | 0.9444 | 0.9067 | 0.9444 | 0.0015 |
| rrf | 0.7222 | 0.8519 | 0.4889 | 0.9444 | 0.8608 | 0.9444 | 0.0140 |
| rrf_reranker | 0.8333 | 0.9074 | 0.5111 | 1.0000 | 0.9050 | 1.0000 | 0.7216 |

Relative to dense-only retrieval, the full reranked stack improved Hit@1 by 0.1667, MRR@5 by 0.1111, and Recall@5 by 0.0556. The cost was substantially higher retrieval latency: +0.7100 seconds/query in this CPU benchmark.

## 3. Embedding model comparison

| Embedding | Dim | Hit@1 | MRR@5 | Recall@5 | nDCG@5 | Page Recall@5 | Indexing (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_minilm_l6_v2 | 384 | 0.8333 | 0.9074 | 1.0000 | 0.9050 | 1.0000 | 6.5853 |
| multi_qa_minilm_l6_cos_v1 | 384 | 0.8333 | 0.9074 | 1.0000 | 0.9050 | 1.0000 | 7.2882 |
| all_mpnet_base_v2 | 768 | 0.8333 | 0.9074 | 1.0000 | 0.9050 | 1.0000 | 52.8556 |

All three tested embedding models produced the same final retrieval-quality metrics under the reranked architecture. The production all-MiniLM-L6-v2 model therefore remains the lowest-change choice. all-mpnet-base-v2 doubled vector dimension from 384 to 768 without a measured quality gain. The first-run model-load timings for new Hugging Face models include download overhead and should not be treated as steady-state load benchmarks.

## 4. Final context Top-K

| Top-K | Hit@K | MRR@K | Precision@K | Recall@K | nDCG@K | Page Recall@K | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1.0000 | 0.9074 | 0.7222 | 0.9722 | 0.9213 | 0.9722 | 0.7872 |
| 5 | 1.0000 | 0.9074 | 0.5111 | 1.0000 | 0.9050 | 1.0000 | 0.7602 |
| 10 | 1.0000 | 0.9074 | 0.3167 | 1.0000 | 0.8894 | 1.0000 | 0.7613 |
| 20 | 1.0000 | 0.9074 | 0.1778 | 1.0000 | 0.8645 | 1.0000 | 0.7710 |

Moving from top_k=3 to top_k=5 increased mean Recall@K by 0.0278 to 1.0000, and Page Recall@K reached 1.0000. Increasing context to 10 or 20 did not improve recall, while precision fell to 0.3167 and 0.1778, respectively.

## 5. Abstention threshold calibration

| Threshold | Accuracy | Answer F1 | Abstention F1 | False answers | False abstentions |
| --- | --- | --- | --- | --- | --- |
| -10.0 | 0.9000 | 0.9231 | 0.8571 | 3 | 0 |
| -8.0 | 0.9333 | 0.9474 | 0.9091 | 2 | 0 |
| -5.0 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 0.0 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 1.0 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 2.0 | 0.9667 | 0.9714 | 0.9600 | 0 | 1 |
| 4.0 | 0.9333 | 0.9412 | 0.9231 | 0 | 2 |
| 6.0 | 0.8333 | 0.8387 | 0.8276 | 0 | 5 |
| 8.0 | 0.4333 | 0.1053 | 0.5854 | 0 | 17 |

At the production threshold 0.0, the 30-case evaluation set had accuracy=1.0000, answer F1=1.0000, and abstention F1=1.0000, with 0 false answers and 0 false abstentions. The observed answerable/unanswerable score separation gap was 8.9097.

## 6. Error and boundary analysis

| Type | Case | Should answer | Value | Detail |
| --- | --- | --- | --- | --- |
| top_k_3_incomplete_retrieval | cash_flow_001 | True | 0.5000 | top_k=3 did not retain all labeled relevant evidence; top_k=5 reached full aggregate recall. |
| confidence_boundary_answerable | report_review_001 | True | 1.3354 | Lowest answerable rank-1 cross-encoder score; closest answerable case to the production threshold. |
| confidence_boundary_unanswerable | unans_010 | False | -7.5743 | Highest unanswerable rank-1 cross-encoder score; closest unanswerable case to the production threshold. |
| production_threshold_errors |  |  | 0.0000 | No answerability classification errors were observed at threshold 0.0 on the 30-case evaluation set. |

No answerability classification errors were observed at threshold 0.0 on this evaluation set. The useful remaining error analysis is therefore boundary-focused: identify the lowest-scoring answerable case, the highest-scoring unanswerable case, and any cases where top_k=3 failed to retain all labeled relevant evidence.

## 7. Baseline vs improved system

| Metric | Dense baseline | Final reranked | Delta |
| --- | --- | --- | --- |
| Hit@1 | 0.6667 | 0.8333 | +0.1667 |
| MRR@5 | 0.7963 | 0.9074 | +0.1111 |
| Recall@5 | 0.9444 | 1.0000 | +0.0556 |
| Mean retrieval latency (s) | 0.0116 | 0.7216 | +0.7100 |

## 8. Interpretation and limitations

- Retrieval conclusions are based on the current 30-case labeled evaluation set (18 answerable, 12 unanswerable) and the sample document.
- Page labels are used when explicit chunk qrels are not available, so Precision/Recall/nDCG should be described as page-labeled retrieval metrics rather than exhaustive chunk-level judgments.
- The perfect threshold-0 answerability result is strong evidence for this benchmark but should not be assumed to generalize to unseen document domains without further calibration.
- Cross-encoder reranking provides the strongest measured retrieval completeness/ranking performance, but it is also the dominant retrieval-latency cost on CPU.
- The local LLM-as-judge quality metrics from Phase 21 remain useful but are not ground-truth human judgments.

## 9. Phase 22 conclusion

The experiment matrix supports the existing production configuration: fixed 200/40 chunking, all-MiniLM-L6-v2 embeddings, hybrid dense+BM25 retrieval, RRF fusion, cross-encoder reranking, top_k=5, and confidence threshold 0.0. The main measured trade-off is retrieval latency from the reranker in exchange for stronger first-result ranking and complete labeled recall.
