import { describe, expect, it } from "vitest";

import { closestSentence, pageLabel, splitCitations, splitSentences } from "@/lib/citations";
import { inventedAnswers, quality, results } from "@/lib/results";
import { getSessionId, resetSessionId } from "@/lib/session";
import { parseEventBlock, readEvents } from "@/lib/sse";
import { percent } from "@/lib/utils";

function streamOf(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)));
      controller.close();
    },
  });
}

async function collect(stream: ReadableStream<Uint8Array>) {
  const events = [];
  for await (const event of readEvents(stream)) events.push(event);
  return events;
}

describe("server-sent events", () => {
  it("parses one block", () => {
    expect(parseEventBlock('event: token\ndata: {"text":"Hi"}')).toEqual({
      event: "token",
      data: { text: "Hi" },
    });
    expect(parseEventBlock("event: ping")).toBeNull();
    expect(parseEventBlock("data: not json")).toBeNull();
  });

  it("reassembles events split across network chunks and CRLF line endings", async () => {
    const events = await collect(
      streamOf(
        'event: token\r\ndata: {"text":"Mea',
        'ls "}\r\n\r\nevent: token\ndata: {"text":"[1]"}\n\nevent: fin',
        'al\ndata: {"answer":"Meals [1]"}\n\n',
      ),
    );
    expect(events.map((e) => e.event)).toEqual(["token", "token", "final"]);
    expect(events[0].data).toEqual({ text: "Meals " });
    expect(events[2].data).toEqual({ answer: "Meals [1]" });
  });
});

describe("citations", () => {
  it("splits answers into text and citation parts", () => {
    expect(splitCitations("Capped at 60 [1][2].")).toEqual([
      { kind: "text", text: "Capped at 60 " },
      { kind: "cite", id: 1 },
      { kind: "cite", id: 2 },
      { kind: "text", text: "." },
    ]);
    expect(splitCitations("No citations")).toEqual([{ kind: "text", text: "No citations" }]);
  });

  it("finds the source sentence that best matches the answer", () => {
    const sentences = splitSentences(
      "Employees work flexibly. The standard working week is 38 hours. Overtime needs approval.",
    );
    expect(sentences).toHaveLength(3);
    expect(closestSentence("The standard working week is 38 hours [1].", sentences)).toBe(1);
    expect(closestSentence("Completely unrelated", sentences)).toBe(-1);
  });

  it("labels pages", () => {
    expect(pageLabel(null, null)).toBeNull();
    expect(pageLabel(3, 3)).toBe("page 3");
    expect(pageLabel(3, 5)).toBe("pages 3–5");
  });
});

describe("session id", () => {
  it("is created once, persisted, and replaced on reset", () => {
    const first = getSessionId();
    expect(first).toMatch(/^[a-f0-9]{48}$/);
    expect(getSessionId()).toBe(first);
    const second = resetSessionId();
    expect(second).not.toBe(first);
    expect(getSessionId()).toBe(second);
  });

  it("replaces an invalid stored value", () => {
    window.localStorage.setItem("docqna_session_id", "bad id!");
    expect(getSessionId()).toMatch(/^[a-f0-9]{48}$/);
  });
});

describe("measured results", () => {
  it("match the documented evaluation", () => {
    expect(percent(quality("ollama").public.correct)).toBe("69.5%");
    expect(percent(quality("gemini").public.correct)).toBe("98.3%");
    expect(inventedAnswers(quality("gemini"))).toEqual({ count: 0, total: 28 });
    expect(inventedAnswers(quality("ollama"))).toEqual({ count: 0, total: 28 });
    expect(results.public_benchmark.hit_at_5).toBe(1);
  });

  it("formats percentages", () => {
    expect(percent(1)).toBe("100%");
    expect(percent(0.3729)).toBe("37.3%");
    expect(percent(null)).toBe("–");
  });
});
