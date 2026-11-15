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
| `compare_pipeline_layers.py` | What does each layer add? Plain RAG, then + keyword search, + reranker, + retrieval gate, + grounding check | `pipeline_layers/phi3_mini/`, `pipeline_layers/gemini/` |
| `attack_prompt_injection.py` | Can a poisoned document or a malicious question take over the answer, before and after the injection defence? | `prompt_injection/phi3_mini/`, `prompt_injection/gemini/` |
| `measure_speed_and_cost.py` | How long does each stage take, and what does an answer cost in tokens and dollars? | `speed_and_cost/phi3_mini/`, `speed_and_cost/gemini/` |
| `build_heldout_set.py` | Builds the held-out test set: 100 questions from 5 SQuAD 2.0 articles, drawn with a fixed seed | writes `data/evaluation/heldout/` |
| `evaluate_heldout.py` | How does the unchanged system do on questions it was never tuned on? | `heldout/phi3_mini/`, `heldout/gemini/` |
| `grading_check.py` | Does the automatic grading agree with a person? Draws 80 shown answers, serves a local blind labelling page, compares the labels | `grading_check/` |

The first five scripts use the private test set, `data/evaluation/private/cases.json`
(30 labelled questions: 18 answerable, 12 unanswerable). Its source document
must be available locally as `data/samples/sample.pdf`. `calibrate_answerability.py` and
`compare_pipeline_layers.py` also use the public test set in `data/evaluation/public/`;
`attack_prompt_injection.py` uses it together with the attack set in `data/evaluation/attacks/`.
`build_heldout_set.py` needs SQuAD 2.0 `dev-v2.0.json` (CC BY-SA 4.0) from
https://rajpurkar.github.io/SQuAD-explorer/ in `data/evaluation/heldout/raw/` (git-ignored);
the sampled set it writes is committed, so `evaluate_heldout.py` runs without it.

Run any script from the repository root, for example:

```bash
python -m experiments.compare_retrieval_methods
```

Change a default in `settings.py` only when a new experiment run justifies it,
and commit the new results with the change.

Tests for the scripts' logic live in `tests/experiments/`, one file per script
(`test_<script name>.py`).
