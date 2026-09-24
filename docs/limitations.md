# DocQ&A Limitations

---

## Current limitations

### No OCR

Image-only or scanned PDFs are not supported. The current PDF path expects extractable text.

### Session isolation is not authentication

DocQ&A isolates document access using a high-entropy anonymous session identifier. The identifier is persisted in a browser cookie so the same browser can retain its anonymous namespace across normal page refreshes.

This does not provide accounts, passwords, OAuth, account recovery, durable cross-device identity, or verified user ownership. A different browser profile or incognito session receives a different anonymous namespace.

### Dataset-specific answerability calibration

The rank-1 cross-encoder threshold of `0` is supported by the current 30-case evaluation dataset and should be recalibrated when the domain, reranker, chunking, or retrieval behavior changes materially.

### Over-citation

The current `phi3:mini` generation behavior can sometimes cite more retrieved sources than are strictly necessary.

### Local-model quality ceiling

The project currently uses Ollama `phi3:mini`, so final answer quality is constrained by that model's capability.

### CPU-oriented latency

Embeddings and cross-encoder reranking are configured for CPU, and the current deployment does not require a GPU. First use can be noticeably slower because Hugging Face models may need to download before they are cached. Local CPU generation with `phi3:mini` can also add response latency.

### No large-scale multi-tenant benchmark yet

Session isolation has automated and manual validation, but the system has not been load-tested as a large multi-user service.

### Public-traffic controls are incomplete

Uploads are capped at 20 MB and the backend enforces the limit. Broader public exposure still needs controls such as:

- request/rate limiting
- storage quotas or automated cleanup
- CPU/memory resource limits
- production timeout policies
- abuse monitoring appropriate to the hosting environment

### Backup and retention policy is not finalized

Docker volumes persist Chroma indexes, registry metadata, Ollama model data, and Hugging Face model downloads across container recreation. A production backup/recovery policy and uploaded-document retention policy have not yet been finalized.

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
