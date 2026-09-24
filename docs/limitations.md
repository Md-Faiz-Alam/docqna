# DocQ&A Limitations

## Current limitations

### No OCR

Image-only/scanned PDFs are not supported. The current PDF path expects extractable text.

### Session isolation is not authentication

DocQ&A isolates document access using a high-entropy anonymous session identifier. It does not provide accounts, passwords, OAuth, account recovery, or durable user identity.

### Dataset-specific answerability calibration

The CE threshold of `0` is supported by the current evaluation dataset and should be recalibrated when the domain, reranker, chunking, or retrieval behavior changes materially.

### Over-citation

The current Phi-3 generation behavior can sometimes cite more retrieved sources than are strictly necessary.

### Local-model quality ceiling

The project currently uses `phi3:mini`, so final answer quality is constrained by that model's capability.

### No large-scale multi-tenant benchmark yet

Session isolation has functional and manual validation, but the system has not yet been load-tested as a large multi-user service.

### Public-deployment safeguards are not complete yet

Before broad public exposure, add practical controls such as upload-size limits, request/rate limits, storage quotas or cleanup, resource limits, timeout policies, and persistent-volume backup strategy.

### Docker deployment is not complete yet

Containerization is the next project phase. Docker support should not be claimed until the containerized stack is validated end to end.

## Non-goals

The current system is not intended to be an autonomous agent framework, GraphRAG implementation, general-purpose web search engine, document editor, or user-account platform.
