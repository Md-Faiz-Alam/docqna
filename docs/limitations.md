# DocQ&A Limitations

---

## Current limitations

### OCR is basic

Scanned pages (pages without a text layer) are rendered and read with Tesseract, limited to `MAX_OCR_PAGES` pages per document. Tesseract is installed in the Docker image; a local run without Tesseract on PATH rejects scanned PDFs with a clear message. Tables, forms and multi-column layouts are flattened to plain text, and handwriting is not supported.

### Some answerable questions still end in abstention

With the phase-23 configuration (retrieval gate `-10.5` plus sentence-level grounding check `3.3`), the public benchmark answers 69.5% of paraphrased answerable questions correctly. Most of the rest end in the safe abstention, not in a wrong answer. Users can still see "not enough information" for questions the documents do answer, especially for paraphrases far from the document's wording.

### Session isolation is not authentication

DocQ&A isolates document access using a high-entropy anonymous session identifier. The identifier is persisted in a browser cookie so the same browser can retain its anonymous namespace across normal page refreshes.

This does not provide accounts, passwords, OAuth, account recovery, durable cross-device identity, or verified user ownership. A different browser profile or incognito session receives a different anonymous namespace.

### Dataset-specific answerability calibration

The retrieval gate (`-10.5`) and grounding threshold (`3.3`) were selected on 105 questions (75 public, 30 private). The evaluation includes only 28 unanswerable questions, so "0 false answers" means "none observed in 28", not a guarantee. Long answers that combine several facts can score low against any single sentence and be withheld. Re-run phase 23 when the domain, LLM, reranker or chunking changes materially.

### Over-citation

The current `phi3:mini` generation behavior can sometimes cite more retrieved sources than are strictly necessary.

### Local-model quality ceiling

The project currently uses Ollama `phi3:mini`, so final answer quality is constrained by that model's capability.

### CPU-oriented latency

Embeddings and cross-encoder reranking are configured for CPU, and the current deployment does not require a GPU. First use can be noticeably slower because Hugging Face models may need to download before they are cached. Local CPU generation with `phi3:mini` can also add response latency.

### No large-scale multi-tenant benchmark yet

Session isolation has automated and manual validation, but the system has not been load-tested as a large multi-user service.

### Public-traffic controls are only partially complete

The backend now enforces a 20 MB upload cap, a 2,000-character question limit, three logical documents per anonymous session, 25 active logical documents globally, 1,000 chunks per document, and 24-hour anonymous-document cleanup. Docker also applies memory caps and log rotation.

Broader public exposure still needs:

- reverse-proxy rate limiting when running more than one API process (the built-in limiter is per process)
- production timeout policies
- host firewall/network rules
- abuse monitoring appropriate to the hosting environment

### Backup policy is not finalized

Docker volumes persist Chroma indexes, registry metadata, Ollama model data, and Hugging Face model downloads across container recreation. Anonymous uploaded documents now expire after 24 hours, but a production backup/recovery policy for persistent volumes has not yet been finalized.

### Public cloud deployment is pending

The Docker stack has been validated locally end to end, including persistence across API-container recreation and browser-refresh session continuity.

The remaining deployment work is to run the validated stack on a remote server and add the reverse-proxy/HTTPS configuration required for an internet-facing public URL.

---

## Non-goals

The current system is not intended to be:

- an autonomous agent framework
- a GraphRAG implementation
- a general-purpose web search engine
- a document editor
- a user-account platform
