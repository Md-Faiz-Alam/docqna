# DocQ&A Security

---

## 1. Scope

DocQ&A processes user-uploaded PDF, DOCX, and TXT documents and sends retrieved document text to the configured LLM for grounded answer generation: a local Ollama model by default, or Google's hosted Gemini API when `LLM_PROVIDER=gemini`.

This document describes the current application security model and the controls that remain before broad public exposure.

---

## 2. Trust boundaries

The main trust boundaries are:

- uploaded document content is untrusted
- browser-provided anonymous session identifiers are untrusted input and are validated by the backend
- `.env` and deployment credentials are operator-managed secrets
- persistent document/index data must be protected by the host and deployment environment
- Ollama is an internal generation service in the Docker topology, not a public API surface
- browser cross-origin access is limited to the `CORS_ALLOWED_ORIGINS` allow-list (default: none); CORS is a browser protection, not authentication

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
- an injection filter that removes text addressed to an AI assistant (instructions to AI systems, attempts to override or reveal the rules, fake prompt sections, Markdown images) from retrieved passages before the LLM sees them; removed text is shown as `[removed: text addressed to an AI assistant]` in the sources
- a link check that withholds any answer containing a link or e-mail address that none of the sources contains
- signed conversation history: each answer carries an HMAC of the session ID and answer text, and assistant turns sent back without a matching signature are dropped, so a client cannot invent assistant turns to steer the follow-up rewriter
- an optional prompt layout (`SEPARATE_PROMPT_ROLES`, off until measured) that puts the answer rules in the LLM's system role and wraps each document in a tag with a random per-request name, so a document cannot close its own tag

Prompt hardening alone was not enough: in the attack test (`experiments/attack_prompt_injection.py`, results in `docs/evaluation.md` section 16) Gemini followed hidden document instructions in 6 of 20 attacks, adding a phishing link and stating a planted false fact, and the grounding check let these through because the attacker's text really was in the cited source. The filter and link check are pattern rules for known attack shapes. They stopped every attack in that test, but they cannot guarantee that a new phrasing is caught.

These controls reduce prompt-injection and unsupported-answer risk, but they do not make arbitrary uploaded content inherently trustworthy. `/qa/stream` sends no draft text by default: the answer leaves the server only after the citation, grounding and link checks have passed, so a withdrawn answer or injected link is never shown, even briefly. `STREAM_DRAFTS=true` turns draft tokens back on (for example for a private local setup where waiting for `phi3:mini` is the bigger problem). It should stay off for any public deployment.

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

FastAPI performs the authoritative size check while streaming the uploaded file in chunks. The website checks the same limit before uploading, but the backend check is the one that counts.

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

On first visit, the website generates a high-entropy random session identifier and keeps it in the browser's `localStorage`, so normal page refreshes reuse the same anonymous namespace. It is never sent to any host other than the configured API.

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

The validated Docker stack uses Ollama only and needs no external API key.

Gemini mode needs `GEMINI_API_KEY`. Keep it only in `.env` or the hosting platform's secret store, never in code or Git. The key is sent only as a request header to Google, and DocQ&A never logs it or puts it in error messages. Gemini mode sends questions and retrieved document passages to Google; on the free tier Google may use them to improve its products. Use Ollama for private or sensitive documents.

---

## 7. Local Docker network exposure

The validated local Compose topology exposes:

```text
FastAPI:   127.0.0.1:8000
```

FastAPI is bound to the Docker host loopback address in the local Compose configuration, and CORS allows only the local website origin by default. A public deployment should expose only the reverse proxy on ports 80/443.

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

Local Docker end-to-end and persistence validation is complete. The public demo runs on Azure Container Apps, which provides HTTPS/TLS at its ingress; the API applies per-session and per-IP rate limits (docs/api.md).

Before broader public exposure, add or verify:

- rate limiting shared across instances (the built-in limiter is per process), if more than one replica runs
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
