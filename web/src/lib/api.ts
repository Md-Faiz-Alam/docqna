import { readEvents } from "@/lib/sse";
import type { ChatTurn, DocumentList, Info, QAResponse } from "@/lib/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);
export const SESSION_HEADER = "X-DocQNA-Session-ID";

/** An API failure with a message that is safe to show to the visitor. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function friendlyMessage(status: number, code: string | undefined, detail: string): string {
  if (code === "llm_rate_limited") return "The answer service is busy right now. Please try again in a minute.";
  if (status === 429) return detail || "Too many requests. Please wait a moment and try again.";
  if (status === 503) return "The answer service is temporarily unavailable. Please try again shortly.";
  if (status >= 500) return "Something went wrong on the server. Please try again.";
  return detail || `Request failed (${status}).`;
}

async function toApiError(response: Response): Promise<ApiError> {
  let code: string | undefined;
  let detail = "";
  try {
    const body = (await response.json()) as { error?: string; detail?: unknown };
    code = body.error;
    detail = typeof body.detail === "string" ? body.detail : "";
  } catch {
    // not JSON
  }
  return new ApiError(friendlyMessage(response.status, code, detail), response.status, code);
}

async function request<T>(
  path: string,
  init: RequestInit & { sessionId?: string; timeoutMs?: number } = {},
): Promise<T> {
  const { sessionId, timeoutMs = 60_000, headers, ...rest } = init;
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...rest,
      headers: { ...(sessionId ? { [SESSION_HEADER]: sessionId } : {}), ...headers },
      signal: rest.signal ?? AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("Could not reach the server.", 0, "network");
  }
  if (!response.ok) throw await toApiError(response);
  return (await response.json()) as T;
}

export const api = {
  health: (timeoutMs = 8_000) => request<{ status: string }>("/health", { timeoutMs }),
  info: () => request<Info>("/info"),
  listDocuments: (sessionId: string) => request<DocumentList>("/documents", { sessionId }),
  loadSamples: (sessionId: string) =>
    request<DocumentList>("/documents/samples", { method: "POST", sessionId, timeoutMs: 120_000 }),
  upload: (sessionId: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<unknown>("/documents", {
      method: "POST",
      body: form,
      sessionId,
      timeoutMs: 300_000,
    });
  },
  deleteDocument: (sessionId: string, documentId: string) =>
    request<unknown>(`/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
      sessionId,
    }),
  feedback: (
    sessionId: string,
    body: {
      rating: "up" | "down";
      question: string;
      answer: string;
      abstained: boolean;
      cited_source_ids: number[];
    },
  ) =>
    request<{ feedback_id: number }>("/feedback", {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

/**
 * Ask a question and stream the answer. `onToken` receives the draft text
 * as it is generated; the resolved value is the final, citation-checked
 * answer, which replaces the draft.
 */
export async function askStream(
  sessionId: string,
  question: string,
  history: ChatTurn[],
  onToken: (text: string) => void,
  signal?: AbortSignal,
): Promise<QAResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/qa/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        [SESSION_HEADER]: sessionId,
      },
      body: JSON.stringify({ question, history }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError("Could not reach the server.", 0, "network");
  }
  if (!response.ok || !response.body) throw await toApiError(response);

  for await (const { event, data } of readEvents(response.body)) {
    if (event === "token") onToken((data as { text: string }).text);
    else if (event === "final") return data as QAResponse;
    else if (event === "error") {
      const { error, detail } = data as { error?: string; detail?: string };
      const status = error === "llm_rate_limited" || error?.startsWith("llm_") ? 503 : 500;
      throw new ApiError(friendlyMessage(status, error, detail ?? ""), status, error);
    }
  }
  throw new ApiError("The answer stream ended unexpectedly.", 0, "stream_ended");
}
