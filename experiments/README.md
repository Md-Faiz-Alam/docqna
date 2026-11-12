# Experiments

Reproducible experiments behind DocQ&A's production RAG configuration.
Each runner evaluates one design decision on
`data/evaluation/evaluation_dataset_v2.json` (30 labelled cases: 18 answerable,
12 unanswerable) and writes results to `data/evaluation/results/`.

| Runner | Decision evaluated | Output |
|---|---|---|
| `run_phase22b_chunking_experiment.py` | Fixed vs structure-aware vs semantic chunking | `phase22b_*` |
| `run_phase22c_retrieval_experiment.py` | Dense vs BM25 vs RRF vs RRF + cross-encoder | `phase22c_*` |
| `run_phase22d_embedding_experiment.py` | Embedding model comparison | `phase22d_*` |
| `run_phase22e_topk_experiment.py` | Final context size (top-k) | `phase22e_*` |
| `run_phase22f_abstention_threshold_experiment.py` | Answerability threshold calibration | `phase22f_*` |
| `run_phase22g_final_analysis.py` | Cross-experiment analysis, charts, manifest | `phase22g_final_analysis/` |

Run from the repository root (the source document referenced by the dataset
must be available locally under `data/samples/`):

```bash
python -m experiments.run_phase22c_retrieval_experiment
```

The selected values are the defaults in `src/docqna/settings.py`. Change them
only when a new experiment run justifies it, and commit the new results with
the change.

Unit tests for the runners' pure logic live in `tests/experiments/`.
