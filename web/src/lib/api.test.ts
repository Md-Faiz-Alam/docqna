import { afterEach, describe, expect, it, vi } from "vitest";

import { API_URL, ApiError, SESSION_HEADER, api, askStream } from "@/lib/api";

const SESSION = "s".repeat(40);

function sse(...events: [string, unknown][]): Response {
  const body = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("askStream", () => {
  it("streams tokens, then resolves with the final answer", async () => {
    const final = { answer: "38 hours [1].", abstained: false, cited_source_ids: [1], sources: [], standalone_question: null };
    const fetchMock = vi.fn().mockResolvedValue(sse(["token", { text: "38 " }], ["token", { text: "hours" }], ["final", final]));
    vi.stubGlobal("fetch", fetchMock);
    const tokens: string[] = [];

    const result = await askStream(SESSION, "Hours?", [{ role: "user", content: "Hi" }], (t) => tokens.push(t));

    expect(result).toEqual(final);
    expect(tokens).toEqual(["38 ", "hours"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/qa/stream`);
    expect(init.headers[SESSION_HEADER]).toBe(SESSION);
    expect(JSON.parse(init.body)).toEqual({ question: "Hours?", history: [{ role: "user", content: "Hi" }] });
  });

  it("turns a stream error event into a friendly ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(sse(["error", { error: "llm_rate_limited", detail: "quota" }])));

    await expect(askStream(SESSION, "q", [], () => undefined)).rejects.toMatchObject({
      name: "ApiError",
      code: "llm_rate_limited",
      message: expect.stringMatching(/busy/),
    });
  });

  it("reports an unreachable server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(askStream(SESSION, "q", [], () => undefined)).rejects.toMatchObject({ code: "network" });
  });
});

describe("api", () => {
  it("shows the server's message for client errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(json(429, { error: "resource_limit", detail: "Session document limit reached (3)." })),
    );

    const error = await api.upload(SESSION, new File(["x"], "a.txt")).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toBe("Session document limit reached (3).");
  });

  it("hides server internals for 5xx errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(500, { error: "indexing_error", detail: "Traceback ..." })));

    await expect(api.listDocuments(SESSION)).rejects.toMatchObject({
      message: "Something went wrong on the server. Please try again.",
    });
  });

  it("sends the session header and posts to the samples endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(json(200, { count: 0, documents: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await api.loadSamples(SESSION);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`${API_URL}/documents/samples`);
    expect(init.method).toBe("POST");
    expect(init.headers[SESSION_HEADER]).toBe(SESSION);
  });
});
