# DocQ&A Roadmap: From Local Project to Public Product

This document explains, in plain language, what DocQ&A is, what has already been built, and the step-by-step plan to turn it into a public website that anyone can use. You do not need to know the code to follow it.

---

## 1. What is DocQ&A? (in one minute)

DocQ&A lets you **upload your own documents** (PDF, Word, or text files) and **ask questions about them**. It answers in plain language and shows exactly **which passage of which document** each answer came from.

Its key promise is **trust**:

- If the documents contain the answer, it answers and **cites the source**.
- If they do not, it **says so honestly** instead of making something up.

Example:

> **You:** How many vacation days do full-time employees get?
> **DocQ&A:** Full-time employees receive 24 days of paid annual leave per year **[1]**.
> *[1] employee_handbook.pdf, page 3*
>
> **You:** Who won the 2018 World Cup?
> **DocQ&A:** The provided documents do not contain enough information to answer the question.

### How it works (simple version)

DocQ&A uses a technique called **RAG** (Retrieval-Augmented Generation). Think of it as an open-book exam:

1. **Read and index:** uploaded documents are split into small passages and stored in a searchable index.
2. **Find (retrieval):** for each question, it searches for the most relevant passages, using both meaning and keywords.
3. **Rank:** a second model re-orders those passages so the best evidence comes first.
4. **Check confidence:** if nothing relevant was found, it stops and says "not enough information".
5. **Write (generation):** an AI language model writes a short answer **using only those passages**, with citation numbers.
6. **Verify:** DocQ&A checks that every citation is valid **and** that the cited passage really supports the answer. If not, it refuses rather than risk a wrong answer.

Steps 1–4 and 6 are DocQ&A's own code. Only step 5 uses an AI model, and that model can be swapped.

---

## 2. Where the project stands today

| Area | Status |
|---|---|
| Upload PDF / Word / text, including **scanned PDFs (OCR)** | ✅ Done |
| Smart search (meaning + keywords) and re-ranking | ✅ Done |
| Answers with citations, plus honest refusal | ✅ Done |
| Answer verification (grounding check) | ✅ Done |
| Follow-up questions ("and what about flights?") | ✅ Done |
| Live "typing" answers (streaming) | ✅ Done |
| 👍/👎 feedback on answers | ✅ Done |
| Privacy between users (each browser sees only its own files) | ✅ Done |
| Abuse protection (file-size, question, and rate limits) | ✅ Done |
| Automated tests (129) and quality checks on every code change | ✅ Done |
| Measured quality on a public 75-question benchmark | ✅ Done |
| Runs on your own computer with Docker | ✅ Done |
| **Public website anyone can open** | ❌ Not yet: this roadmap |
| **Modern, product-quality user interface** | ❌ Not yet: this roadmap |

### Measured quality

On a public test set of 75 questions (59 answerable, 16 deliberately unanswerable):

| What we measure | Local model (`phi3:mini`) | Gemini |
|---|---|---|
| The right passage is found in the top 5 results | **100%** | **100%** (same search) |
| Answerable questions answered correctly | **69.5%** | **98.3%** |
| Made-up answers to unanswerable questions (both test sets) | **0 of 28** | **0 of 28** |
| Typical time per answer | slow on a normal computer | **about 2 seconds** |

The remaining answerable questions end in an honest "not enough information", not a wrong answer.

---

## 3. The goal

> A **public website** where anyone, from anywhere, can open a link and try DocQ&A end to end (upload a document, ask questions, see sources) **without the owner's computer being switched on**, built and hosted **completely free**.

At the same time, keep a **private mode** for people who want to run it on their own machine, where no data ever leaves their computer.

---

## 4. The target setup

```text
          Anyone's browser (phone or laptop)
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  WEBSITE (frontend)                               │
│  Modern Next.js site, hosted free on Vercel       │
│  • Landing page  • Chat app  • Results page       │
└───────────────────────────────────────────────────┘
                        │  secure internet connection
                        ▼
┌───────────────────────────────────────────────────┐
│  BACKEND (the "brain")                            │
│  DocQ&A API, hosted free on Hugging Face Spaces   │
│  • reads documents  • searches  • ranks  • checks │
└───────────────────────────────────────────────────┘
                        │  only the 5 best passages
                        ▼
┌───────────────────────────────────────────────────┐
│  AI WRITER (language model)                       │
│  Public demo: Google Gemini (free tier)           │
│  Private mode: Ollama, running locally            │
└───────────────────────────────────────────────────┘
```

### Two modes

| | **Public demo** | **Private / self-hosted** |
|---|---|---|
| Who uses it | Anyone with the link | The owner, or a company on its own server |
| Where it runs | Vercel + Hugging Face (cloud) | Your own computer or server (Docker) |
| AI model | Google Gemini (fast, strong) | Ollama `phi3:mini` (local) |
| Speed | A few seconds per answer | About 45 seconds per answer on a normal PC |
| Privacy | Passages are sent to Google: **do not upload confidential files** | Nothing leaves the machine |
| Cost | Free | Free |

### Why these services?

| Service | Role | Why it was chosen |
|---|---|---|
| **Vercel** | Hosts the website | Free for personal projects, very fast worldwide, built for Next.js |
| **Hugging Face Spaces** | Runs the backend | Free with 16 GB of memory, enough for the search and ranking models. Vercel, Netlify and Render's free tier are too small for these models |
| **Google Gemini (free tier)** | Writes answers in the demo | Free, fast, and much stronger than small local models |
| **Ollama** | Writes answers in private mode | Runs on your own computer; data stays private |
| **GitHub** | Stores the code; runs automatic checks | Already in use |

---

## 5. The step-by-step plan

Each step has a clear goal and a clear "done when", so progress is easy to check.

### Step 1: Add Google Gemini as an answer writer
- **What:** teach DocQ&A to use Gemini as well as Ollama, chosen with one setting (`LLM_PROVIDER=gemini` or `ollama`).
- **Why:** the free public server is too slow for a local AI model; Gemini answers in seconds.
- **Done when:** both options work, tests pass, and Ollama is still the default for private use.
- **Status:** ✅ Done. Gemini is built and tested with a stand-in for Google's servers; the first real answers need the key from step 2.

### Step 2: Add the Gemini key
- **What:** create a free API key in Google AI Studio, in its own project, and put it in the local `.env` file.
- **Why:** the key is like a password that lets DocQ&A use Gemini. It must never be shared or committed to GitHub.
- **Done when:** the key is in `.env` (which Git ignores).

### Step 3: Re-tune the safety checks for Gemini
- **What:** re-run the 105-question test with Gemini and recalculate the thresholds that decide when to refuse.
- **Why:** the current settings were tuned for the local model; a different AI model behaves differently.
- **Done when:** Gemini gets its own measured, documented settings, plus a side-by-side comparison of the local model and Gemini.
- **Status:** ✅ Done. With Gemini: 98.3% of public questions answered correctly (local model: 69.5%), 100% of private ones, 0 invented answers out of 28, about 2 seconds per answer. Gemini gets its own safety-check setting (1.5 instead of 3.3).

### Step 4: Prepare the backend for the cloud
- **What:**
  - package the backend as one container for Hugging Face;
  - allow only the official website to call the API (CORS);
  - add an `/info` endpoint so the website can show which model is in use;
  - add a "load sample documents" option so visitors can try it instantly.
- **Done when:** the container runs locally exactly as it will in the cloud.
- **Status:** ✅ Done. The cloud container (`docker/cloud.Dockerfile`) was built and run locally with Gemini: sample documents load in about a second, answers come back in about 3 seconds with citations, off-topic questions are refused, only the allowed website can call it from a browser, and it uses about 580 MB of memory.

### Step 5: Build the modern website
- **What:** a new website using industry-standard tools (Next.js, TypeScript, Tailwind CSS, shadcn/ui), with three pages:
  1. **Landing page:** what DocQ&A is, how it works, live quality numbers, and a "Try the demo" button.
  2. **App:** upload by drag and drop, a chat with live answers, clickable citations that open the exact source passage with the supporting sentence highlighted, follow-up questions, feedback buttons, and a friendly "waking up the server" message.
  3. **Results page:** charts of the measured quality and the comparisons.
- **Also:** dark/light mode, mobile-friendly layout, accessibility, automated tests.
- **Done when:** the full flow works in a browser against the backend, and checks pass automatically on GitHub.
- **Status:** ✅ Built in `web/` and tested in a browser against the cloud container: sample documents, streamed answers, clickable citations with the matching sentence highlighted, follow-up questions, feedback, an honest refusal, and the "waking up" message all work. Lint, type checks, 20 tests and the production build pass locally; the GitHub check runs on the next push.

### Step 6: Go live
- **What:**
  1. create free accounts on **Hugging Face** and **Vercel**;
  2. deploy the backend to a Hugging Face Space and add the Gemini key as a **secret** there;
  3. deploy the website to Vercel and point it at the backend.
- **Done when:** the public link works from any device with the owner's computer switched off.

### Step 7: Polish and share
- **What:** a README with a screenshot or GIF and the live link, plus a short write-up of the quality results (for example, *"from 37% to 69.5% correct answers with zero invented answers"*).
- **Done when:** the link is on the CV, LinkedIn and the GitHub page.

### Summary

| Step | What |
|---|---|
| 1 | Gemini support (Ollama stays) ✅ |
| 2 | Create and store the Gemini key |
| 3 | Re-tune safety checks for Gemini ✅ |
| 4 | Cloud-ready backend ✅ |
| 5 | Modern website ✅ |
| 6 | Accounts and deployment |
| 7 | Polish and share |

---

## 6. Good to know (limits of the free setup)

- **Sleeping server:** a free Hugging Face Space sleeps after about 48 hours with no visitors. The next visitor waits about 1–2 minutes while it wakes up; the website shows a friendly message and retries automatically.
- **Uploads are temporary:** uploaded documents are cleared when the server restarts, and automatically after 24 hours. Sample documents are always available.
- **Shared free quota:** all visitors share the free Gemini limits. Under heavy use, some visitors may briefly see "busy, try again shortly". Built-in rate limits stop any single visitor from using it all up.
- **Privacy in the demo:** in the public demo, passages from uploaded files are processed by Google Gemini, and the free tier may be used by Google to improve its products. **Do not upload confidential documents to the demo.** Use private mode for sensitive files.
- **Free-tier rules change:** the free limits of Hugging Face, Vercel and Gemini can change over time. Check their current terms before deploying.

---

## 7. Rules we follow

- **No secrets in code:** API keys live only in `.env` (local) or as platform secrets (cloud).
- **Measure before changing:** quality settings change only when a test run proves the change is better.
- **Honest numbers:** results are published with their caveats; limitations are documented, not hidden.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: find relevant passages first, then let an AI write an answer using only them |
| **LLM** | Large Language Model: the AI that writes text (Gemini, phi3) |
| **Ollama** | Free software that runs AI models on your own computer |
| **Gemini** | Google's AI model, usable through a free API key |
| **API key** | A secret code that lets an app use an online service; treat it like a password |
| **Embedding** | A list of numbers representing a passage's meaning, used for meaning-based search |
| **BM25** | Classic keyword search |
| **Reranker (cross-encoder)** | A model that re-orders search results by how well they answer the question |
| **Citation** | A number such as [1] linking an answer to its source passage |
| **Grounding check** | Verifying that the cited passage really supports the answer |
| **Abstention** | Honestly refusing to answer when the documents do not contain the answer |
| **OCR** | Optical Character Recognition: reading text from scanned images |
| **Streaming** | Showing the answer word by word as it is written |
| **Frontend / backend** | The website people see / the server that does the work |
| **Hugging Face Spaces** | A free cloud service for hosting AI apps |
| **Vercel** | A free cloud service for hosting websites |
| **Docker / container** | A standard package that runs the same way on any computer or server |
| **CI** | Continuous Integration: automatic tests that run on GitHub after every change |
| **Benchmark** | A fixed set of test questions used to measure quality objectively |
