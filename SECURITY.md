# DocQ&A Security

---

## 1. Scope

DocQ&A processes user-uploaded PDF, DOCX, and TXT documents and sends retrieved document text to a configured local Ollama model for grounded answer generation.

This document describes the current application security model and the controls that remain before broad public exposure.

---

## 2. Trust boundaries

The main trust boundaries are:

- uploaded document content is untrusted
- browser-provided anonymous session identifiers are untrusted input and are validated by the backend
- `.env` and deployment credentials are operator-managed secrets
- persistent document/index data must be protected by the host and deployment environment
- Ollama is an internal generation service in the Docker topology, not a public API surface

---

## 3. Untrusted document content

Uploaded document text is treated as untrusted data, not application instruction.

Application instructions, user questions, and retrieved evidence are separated. Document text must not override application-level instructions.

The RAG pipeline includes:

- prompt hardening
- response sanitization
- exact safe abstention
- citation normalization/repair
- citation validation
- safe fallback behavior when citation validation fails

These controls reduce prompt-injection and unsupported-answer risk, but they do not make arbitrary uploaded content inherently trustworthy.

---

## 4. File-upload controls

Supported document types are:

```text
PDF
DOCX
TXT
```

The configured upload limit is:

```text
20 MB
```

FastAPI performs the authoritative size check while streaming the uploaded file in chunks. Streamlit is configured with the same upload limit for the product UI.

Oversized uploads are rejected rather than fully copied into application storage.

Additional public-demo limits are enforced by the backend:

```text
Question length:              2,000 characters
Logical documents/session:   3
Active logical documents:    25
Chunks/document:              1,000
Anonymous document retention: 24 hours
```

Scanned PDF pages are OCR-processed with Tesseract, with a per-document page limit (`MAX_OCR_PAGES`) to bound CPU use.

Answer feedback (`POST /feedback`) stores the question, the answer and an optional comment together with the anonymous session ID, so that weak answers can be reviewed. It is deleted after `FEEDBACK_RETENTION_DAYS` (default 30). Operators who must not keep question text should set a short retention period.

---

## 5. Anonymous session isolation

The application uses:

```text
X-DocQNA-Session-ID
```

to isolate anonymous browser sessions.

On first use, Streamlit generates a high-entropy UUID4-based session identifier. The identifier is persisted in the browser as `docqna_session_id` so normal page refreshes can reuse the same anonymous namespace.

The browser cookie uses `Path=/` and `SameSite=Lax`; the `Secure` attribute is added when the application is served over HTTPS.

The backend validates the session header and enforces session scope across:

- document registration
- duplicate/version checks
- Chroma metadata
- BM25 retrieval
- QA retrieval
- document listing
- document deletion

This mechanism is **not authentication**. It does not establish a verified user identity, account ownership, cross-device identity, or account recovery.

Because possession of a valid anonymous session identifier can identify that session namespace, session IDs should not be logged, exposed, or reused as general credentials.

---

## 6. Secrets and configuration

Secrets and machine-specific configuration belong in `.env`.

`.env` is excluded from Git.

Do not commit:

- API keys
- passwords
- tokens
- private certificates
- production credentials

`.env.example` should contain only safe defaults and placeholders.

The production Docker configuration currently uses Ollama only; no external generation API key is required for the validated stack.

---

## 7. Local Docker network exposure

The validated local Compose topology exposes:

```text
Streamlit: 127.0.0.1:8501
FastAPI:   127.0.0.1:8000
```

FastAPI and Streamlit are both bound to the Docker host loopback address in the local Compose configuration. A public deployment should expose only the reverse proxy on ports 80/443.

Ollama is reachable by FastAPI over the internal Compose network and is not directly published to the host by the Compose file.

This local network posture is not a substitute for a production reverse proxy, firewall policy, or HTTPS configuration.

---

## 8. Persistent data

The Docker stack uses named volumes for:

- Chroma indexes
- document registry metadata
- Ollama model data
- Hugging Face model cache

Container recreation does not remove these volumes.

Do not run:

```bash
docker compose down -v
```

unless deleting persisted Docker data is intentional.

Anonymous uploaded-document metadata and indexes are automatically expired after 24 hours of inactivity based on the latest document version. Public deployment still needs an explicit persistent-volume backup/recovery strategy.

---

## 9. Logging and sensitive content

Avoid logging:

- full source documents
- unnecessary retrieved passages
- anonymous session identifiers
- secrets or credentials

Docker JSON logs are rotated with bounded size/file-count settings. Host-level log retention and monitoring should still be configured for the final deployment environment.

Operators are responsible for appropriate data-retention and hosting policies for uploaded documents.

---

## 10. Public deployment status

Local Docker end-to-end and persistence validation is complete.

Public cloud deployment is still pending. Before broad public exposure, add or verify:

- reverse proxy and HTTPS/TLS
- request/rate limiting
- reverse-proxy request/rate limiting
- production timeout policies
- host-level log-retention/monitoring policy
- persistent-volume backup/recovery
- host firewall/network rules
- monitoring appropriate to the deployment environment

The 20 MB upload-size limit, question/document/chunk quotas, 24-hour anonymous-document cleanup, Docker memory limits, and Docker log rotation are already implemented and are not outstanding deployment controls.

---

## 11. Vulnerability reporting

If the repository is public, prefer GitHub private security reporting when enabled, or contact the repository owner directly.

Do not publish exploit details, credentials, or sensitive document content in a public issue.
