# Experiments

Reproducible experiments behind DocQ&A's configuration. Each script tests one
design decision and writes its results to its own folder under
`data/evaluation/results/`. The selected values are the defaults in
`src/docqna/settings.py`.

| Script | Question it answers | Results folder |
|---|---|---|
| `compare_chunking_strategies.py` | How should documents be split: fixed size, by structure, or by meaning? | `chunking_strategies/` |
| `compare_retrieval_methods.py` | Which search works best: dense, BM25, hybrid (RRF), or hybrid + reranker? | `retrieval_methods/` |
| `compare_embedding_models.py` | Which embedding model finds the right passage most often? | `embedding_models/` |
| `compare_context_sizes.py` | How many passages (top-k) should the answer use? | `context_sizes/` |
| `calibrate_retrieval_threshold.py` | Below which search score should the system refuse? (first, retrieval-only version) | `retrieval_threshold/` |
| `summarize_retrieval_experiments.py` | Combines the five results above into one report, charts and a manifest | `retrieval_summary/` |
| `calibrate_answerability.py` | With the LLM in the loop: which retrieval gate and grounding check give correct answers without invented ones? `--provider gemini` repeats it for Gemini | `answerability/phi3_mini/`, `answerability/gemini/` |

The first five scripts use the private test set, `data/evaluation/private/cases.json`
(30 labelled questions: 18 answerable, 12 unanswerable). Its source document
must be available locally as `data/samples/sample.pdf`. `calibrate_answerability.py`
also uses the public test set in `data/evaluation/public/`.

Run any script from the repository root, for example:

```bash
python -m experiments.compare_retrieval_methods
```

Change a default in `settings.py` only when a new experiment run justifies it,
and commit the new results with the change.

Tests for the scripts' logic live in `tests/experiments/`, one file per script
(`test_<script name>.py`).
