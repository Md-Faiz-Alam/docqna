# DocQ&A website

The public website for DocQ&A: a landing page, the live demo and the measured results. Built with Next.js (App Router), TypeScript, Tailwind CSS and shadcn/ui-style components; charts use Recharts.

| Page | What it does |
|---|---|
| `/` | What DocQ&A is, how it works, and the headline quality numbers |
| `/demo` | Upload by drag and drop (or load the sample documents), chat with streamed answers, click a citation to see the source passage with the closest sentence highlighted, ask follow-ups, rate answers |
| `/results` | Charts and tables of the measured results |

## Run locally

Needs Node.js 20.9 or newer and a running backend (see the main [README](../README.md)).

```bash
cd web
npm ci
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL to your backend
npm run dev                  # http://localhost:3000
```

The backend must allow the website's address: set `CORS_ALLOWED_ORIGINS=http://localhost:3000` on the backend.

## Checks

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

CI runs all four on every push.

## Where the numbers come from

Every number on the site comes from `src/data/results.json`, which is generated from the committed evaluation results. After a new evaluation run, regenerate it from the repository root:

```bash
python -m scripts.export_web_results
```

## Notes

- The visitor's anonymous session ID is kept in `localStorage`; it separates visitors' documents and is not a login. "Start over with an empty workspace" creates a new one.
- A free backend may be asleep. The demo shows "Waking up the server…" and keeps retrying for up to three minutes.
- The highlighted sentence in a source passage is the one that shares the most words with the answer. It helps the reader find the evidence; it is not the backend's grounding score.
