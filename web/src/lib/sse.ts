/** Minimal parser for the backend's server-sent events (`/qa/stream`). */

export interface SseEvent {
  event: string;
  data: unknown;
}

/** Parse one complete event block ("event: x\ndata: {...}"). */
export function parseEventBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

/** Yield events from a streamed response body, however the bytes are split. */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const parsed = parseEventBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
    const rest = parseEventBlock(buffer + decoder.decode());
    if (rest) yield rest;
  } finally {
    reader.releaseLock();
  }
}
