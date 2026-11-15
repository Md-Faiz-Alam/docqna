"use client";

import { ArrowUpIcon, Loader2Icon, RotateCcwIcon, SquareIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { AnswerCard, AnswerText, type Rating } from "@/components/demo/answer";
import { CitationSheet } from "@/components/demo/citation-sheet";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, askStream } from "@/lib/api";
import type { ChatTurn, Citation, QAResponse } from "@/lib/types";

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  response?: QAResponse;
  error?: string;
  rating?: Rating;
  pending?: boolean;
}

const MAX_HISTORY_TURNS = 12;

/** Earlier completed turns, oldest first, in the shape `/qa` accepts. */
export function historyFrom(messages: Message[]): ChatTurn[] {
  return messages
    .filter((m) => !m.pending && !m.error && m.content.trim())
    .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }))
    .slice(-MAX_HISTORY_TURNS);
}

const SUGGESTIONS = [
  "How many hours is the standard working week?",
  "How long is the CloudBox S2 warranty?",
  "What uptime does the cloud SLA guarantee?",
];

export function Chat({
  sessionId,
  maxChars,
  hasDocuments,
  samplesAvailable,
}: {
  sessionId: string;
  maxChars: number;
  hasDocuments: boolean;
  samplesAvailable: boolean;
}) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState<{ citation: Citation; answer: string } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);
  const busy = messages.some((m) => m.pending);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function update(id: number, patch: Partial<Message>) {
    setMessages((all) => all.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy || text.length > maxChars) return;

    const history = historyFrom(messages);
    const userId = nextId.current++;
    const answerId = nextId.current++;
    setMessages((all) => [
      ...all,
      { id: userId, role: "user", content: text },
      { id: answerId, role: "assistant", content: "", pending: true },
    ]);
    setInput("");

    const controller = new AbortController();
    abortRef.current = controller;
    let draft = "";
    try {
      const response = await askStream(
        sessionId,
        text,
        history,
        (token) => {
          draft += token;
          update(answerId, { content: draft });
        },
        controller.signal,
      );
      update(answerId, { content: response.answer, response, pending: false });
    } catch (error) {
      const stopped = error instanceof DOMException && error.name === "AbortError";
      update(answerId, {
        pending: false,
        content: "",
        error: stopped
          ? "Stopped."
          : error instanceof ApiError
            ? error.message
            : "Something went wrong. Please try again.",
      });
    } finally {
      abortRef.current = null;
    }
  }

  function rate(message: Message, rating: Rating) {
    const response = message.response;
    if (!response) return;
    const index = messages.findIndex((m) => m.id === message.id);
    const question = messages[index - 1]?.content ?? "";
    update(message.id, { rating });
    void api
      .feedback(sessionId, {
        rating,
        question,
        answer: response.answer,
        abstained: response.abstained,
        cited_source_ids: response.cited_source_ids,
      })
      .catch(() => undefined);
  }

  const tooLong = input.length > maxChars;

  return (
    <div className="ring-gradient flex min-h-[34rem] flex-col rounded-2xl shadow-xl shadow-primary/5">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="font-semibold">Chat</h2>
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" disabled={busy} onClick={() => setMessages([])}>
            <RotateCcwIcon /> New chat
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-5" aria-live="polite" aria-busy={busy}>
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 py-10 text-center">
            <p className="max-w-sm text-muted-foreground">
              {hasDocuments
                ? "Ask anything about your documents. Follow-up questions work too."
                : samplesAvailable
                  ? "Load the sample documents or upload your own to start."
                  : "Upload a document to start."}
            </p>
            {hasDocuments && (
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <Button key={s} variant="outline" size="sm" onClick={() => void ask(s)}>
                    {s}
                  </Button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <ol className="flex flex-col gap-5">
            {messages.map((message) =>
              message.role === "user" ? (
                <li
                  key={message.id}
                  className="ml-auto max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 whitespace-pre-wrap text-primary-foreground"
                >
                  <span className="sr-only">You: </span>
                  {message.content}
                </li>
              ) : (
                <li key={message.id} className="max-w-[92%] rounded-2xl rounded-bl-sm bg-muted px-4 py-3">
                  <span className="sr-only">DocQ&amp;A: </span>
                  {message.error ? (
                    <p className="text-destructive">{message.error}</p>
                  ) : message.response ? (
                    <AnswerCard
                      response={message.response}
                      rating={message.rating}
                      onCite={(citation) => setOpen({ citation, answer: message.content })}
                      onRate={(rating) => rate(message, rating)}
                    />
                  ) : message.content ? (
                    <div className="opacity-80">
                      <AnswerText text={message.content} sources={[]} onCite={() => undefined} />
                      <p className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                        <Loader2Icon className="size-3 animate-spin" aria-hidden /> Writing and
                        checking citations…
                      </p>
                    </div>
                  ) : (
                    <p className="flex items-center gap-2 text-muted-foreground">
                      <Loader2Icon className="size-4 animate-spin" aria-hidden /> Searching your
                      documents…
                    </p>
                  )}
                </li>
              ),
            )}
          </ol>
        )}
        <div ref={endRef} />
      </div>

      <form
        className="border-t p-3"
        onSubmit={(event) => {
          event.preventDefault();
          void ask(input);
        }}
      >
        <label htmlFor="question" className="sr-only">
          Your question
        </label>
        <div className="flex items-end gap-2">
          <Textarea
            id="question"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                void ask(input);
              }
            }}
            placeholder={hasDocuments ? "Ask a question…" : "Add a document first"}
            disabled={!hasDocuments}
            rows={1}
            className="max-h-40 resize-none"
            aria-invalid={tooLong}
            aria-describedby="question-hint"
          />
          {busy ? (
            <Button type="button" size="icon" variant="outline" aria-label="Stop" onClick={() => abortRef.current?.abort()}>
              <SquareIcon />
            </Button>
          ) : (
            <Button type="submit" size="icon" aria-label="Send" disabled={!hasDocuments || !input.trim() || tooLong}>
              <ArrowUpIcon />
            </Button>
          )}
        </div>
        <p id="question-hint" className="mt-1.5 text-xs text-muted-foreground">
          {tooLong ? (
            <span className="text-destructive">
              {input.length}/{maxChars} characters: please shorten your question.
            </span>
          ) : (
            "Enter to send, Shift+Enter for a new line."
          )}
        </p>
      </form>

      <CitationSheet citation={open?.citation ?? null} answer={open?.answer ?? ""} onClose={() => setOpen(null)} />
    </div>
  );
}
