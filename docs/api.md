# DocQ&A API

The FastAPI backend in `src/docqna/api/`. Interactive docs are served at `/docs`.

```text
GET    /health                    liveness (loads nothing)
GET    /ready                     readiness: vector store, registry, LLM (503 if any is down)
GET    /info                      active LLM, models, sample documents and limits (no secrets)
GET    /metrics                   Prometheus metrics
POST   /qa                        answer with verified citations (JSON)
POST   /qa/stream                 same, streamed as server-sent events
POST   /feedback                  thumbs up/down on an answer
POST   /documents                 upload and index a document
POST   /documents/samples         load the fictional sample documents into the session
GET    /documents                 list the session's documents
DELETE /documents/{document_id}   delete a document (all versions)
```

Document and QA operations require the anonymous session header enforced by the backend. Interactive API docs are at `/docs`.

**Streaming (`/qa/stream`).** The response is `text/event-stream`. It sends exactly one `final` event (the same body as `/qa`) or one `error` event (`{"error", "detail"}`). By default it sends no draft text, because drafts have not yet passed the citation, grounding and link checks, and a draft could show a made-up answer or an injected link that the checks then withdraw. With `STREAM_DRAFTS=true` it also sends `token` events (`{"text": ...}`) while the model generates. In that mode the `final` answer is still authoritative and replaces the draft; it may be a corrected answer or the abstention.

**Conversation memory.** `/qa` and `/qa/stream` accept an optional `history` of up to 12 earlier turns (`{"role": "user"|"assistant", "content": ...}`, oldest first). When history is present, the question is first rewritten into a standalone question ("And for flights?" becomes "Which flight class is allowed for business travel?"). That rewritten question drives retrieval and the answer, and it is returned as `standalone_question`. If the rewrite fails, the original question is used.

Every `/qa` response and `final` stream event includes `answer_signature`. Send it back as the `signature` of that answer's assistant turn. Assistant turns without a valid signature (made up by the client, edited, or from another session) are dropped before the rewrite, so a caller cannot put words in the assistant's mouth; user turns are used as sent. Signatures use `HISTORY_SIGNING_KEY`, or a random key per process when it is unset, so answers from before a restart stop counting as history. Set the key if several instances serve the same users.

**Feedback.** `POST /feedback` stores a rating with the question and answer text, kept for `FEEDBACK_RETENTION_DAYS` (default 30). `python scripts/export_feedback.py --out drafts.json` exports down-voted answers as draft evaluation cases for a person to label.

**Sample documents.** `POST /documents/samples` indexes the four fictional company documents from the public benchmark into the caller's session, so a visitor can try the demo without uploading anything. Calling it again skips samples already loaded. Samples do not count toward the per-session or server document limits.

**CORS.** Browsers may call the API only from the origins in `CORS_ALLOWED_ORIGINS` (comma-separated, for example `https://docqna.vercel.app`), plus any origin matching `CORS_ALLOWED_ORIGIN_REGEX` (for example Vercel preview URLs). The default is no cross-origin access. CORS stops other websites' scripts in a browser; it is not authentication, and non-browser clients can still call the API.

**Rate limits.** By default each session gets 20 questions, 10 uploads (including sample loads) and 30 feedback posts per minute per API process. Over the limit, requests get `429` with `Retry-After`. Change this with `RATE_LIMIT_QA_PER_MINUTE`, `RATE_LIMIT_UPLOADS_PER_MINUTE` and `RATE_LIMIT_FEEDBACK_PER_MINUTE`; `0` disables a limit.

Session IDs are chosen by the browser, so a script could send a new one with every request. Each client IP therefore also gets `RATE_LIMIT_IP_FACTOR` times those limits (default 3, so 60 questions per minute). The factor leaves room for several people behind one address; `0` turns the IP limit off. Behind a proxy, set `TRUSTED_PROXY_HOPS` to the number of proxies that append to `X-Forwarded-For` (Azure Container Apps: 1, set in the cloud image). The limiter then uses the entry the outermost trusted proxy added, never the client-supplied entries to its left. With `0` it uses the socket address and ignores the header. If you run several API instances, also rate-limit at the reverse proxy, because each process keeps its own counters.

**Metrics.** Request count and latency are reported per route template. The QA counters split answered and abstained responses. There are also RAG retrieval and generation latency, BM25 index rebuild, and rate-limit counters. Labels never include document IDs, session IDs, or query text.

## Anonymous session isolation

The application uses a high-entropy anonymous session identifier sent to the API as:

```text
X-DocQNA-Session-ID
```

On first visit, the website generates a random session ID and keeps it in the browser's `localStorage`, so a page refresh stays in the same anonymous document namespace. "Start over with an empty workspace" creates a new one.

The backend owns and enforces session scope across document registration, deduplication/versioning, Chroma metadata filtering, BM25 filtering, QA retrieval, listing, and deletion.

This is **session isolation, not authentication**. A different browser profile or incognito session receives a different anonymous namespace, and there is no account recovery or cross-device identity.
