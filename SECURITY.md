# Security

## Scope

DocQ&A processes user-uploaded documents and sends retrieved document text to a configured Ollama model for grounded answer generation.

## Untrusted document content

Uploaded document text is treated as untrusted data.

Application instructions, user questions, and retrieved evidence are separated. Document text must not override application-level instructions.

The RAG pipeline includes prompt hardening, response sanitization, citation validation, and safe abstention behavior.

## Secrets

Secrets and machine-specific configuration belong in `.env`.

`.env` is excluded from Git.

Do not commit API keys, passwords, tokens, private certificates, or production credentials. `.env.example` should contain only safe defaults/placeholders.

## Anonymous session isolation

The application uses `X-DocQNA-Session-ID` to isolate anonymous sessions.

The backend validates the header and enforces session scope across document lifecycle and retrieval operations.

This is not authentication.

## Public deployment

Before broad public exposure, add:

- upload-size limits
- request/rate limiting
- CPU/memory limits
- storage quotas/cleanup
- TLS/reverse proxy where appropriate
- log-retention policy
- backup/recovery policy for persistent data

## Sensitive document content

Avoid logging full source documents or unnecessary retrieved passages in production.

Operators are responsible for appropriate retention and hosting policies for uploaded documents.

## Reporting a vulnerability

If the repository is public, prefer GitHub private security reporting if enabled, or contact the repository owner directly instead of publishing exploit details in a public issue.
