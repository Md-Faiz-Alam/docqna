"use client";

import { CircleSlashIcon, FileTextIcon, SparklesIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/** Example exchanges, taken from real answers on the sample documents. */
const SCRIPT = [
  {
    question: "How long is the CloudBox S2 warranty?",
    answer: "The enclosure has a three-year limited hardware warranty from the date of purchase",
    cite: "cloudbox_s2_warranty.txt",
  },
  {
    question: "What uptime does the cloud SLA guarantee?",
    answer:
      "At least 99.9% monthly uptime for Business subscriptions and 99.95% for Enterprise subscriptions",
    cite: "northwind_cloud_sla.txt",
  },
  {
    question: "Who won the 2018 World Cup?",
    answer: "The provided documents do not contain enough information to answer the question.",
    cite: null,
  },
] as const;

type Phase = "asking" | "thinking" | "typing" | "done";

function prefersReducedMotion() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** A looping, self-typing example conversation for the hero. */
export function LiveChat() {
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("done");
  const [typed, setTyped] = useState<number>(SCRIPT[0].answer.length);
  const item = SCRIPT[index];

  useEffect(() => {
    if (prefersReducedMotion()) return;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const at = (ms: number, fn: () => void) => timers.push(setTimeout(fn, ms));
    let typing: ReturnType<typeof setInterval> | undefined;

    at(0, () => {
      setPhase("asking");
      setTyped(0);
    });
    at(700, () => setPhase("thinking"));
    at(1700, () => {
      setPhase("typing");
      let n = 0;
      typing = setInterval(() => {
        n += 2;
        setTyped(Math.min(n, item.answer.length));
        if (n >= item.answer.length) {
          clearInterval(typing);
          setPhase("done");
        }
      }, 28);
    });
    at(1700 + item.answer.length * 14 + 2600, () => setIndex((i) => (i + 1) % SCRIPT.length));

    return () => {
      timers.forEach(clearTimeout);
      if (typing) clearInterval(typing);
    };
  }, [index, item.answer.length]);

  const abstained = item.cite === null;

  return (
    <div className="ring-gradient relative rounded-2xl p-1 shadow-2xl shadow-primary/10">
      <div className="rounded-[calc(var(--radius-2xl)-4px)] bg-card">
        <div className="flex items-center gap-2 border-b px-4 py-3">
          <span className="flex gap-1.5" aria-hidden>
            <span className="size-2.5 rounded-full bg-[oklch(0.7_0.17_25)]" />
            <span className="size-2.5 rounded-full bg-[oklch(0.82_0.15_85)]" />
            <span className="size-2.5 rounded-full bg-[oklch(0.75_0.15_150)]" />
          </span>
          <span className="ml-2 text-xs text-muted-foreground">docqna · live example</span>
          <span className="ml-auto flex items-center gap-1 text-[11px] text-success">
            <span className="size-1.5 animate-pulse rounded-full bg-success" aria-hidden /> grounded
          </span>
        </div>

        <div className="flex min-h-64 flex-col gap-4 p-5 text-sm" aria-live="off">
          <div
            key={`q-${index}`}
            className="ml-auto max-w-[85%] animate-in rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-primary-foreground fade-in-0 slide-in-from-bottom-2 duration-300"
          >
            {item.question}
          </div>

          {phase === "asking" ? null : phase === "thinking" ? (
            <div className="flex w-fit items-center gap-2 rounded-2xl rounded-bl-sm bg-muted px-4 py-3 text-muted-foreground">
              <SparklesIcon className="size-3.5 animate-pulse text-primary" aria-hidden />
              Searching 4 documents
              <span className="flex gap-0.5" aria-hidden>
                {[0, 150, 300].map((d) => (
                  <span key={d} className="size-1 animate-bounce rounded-full bg-current" style={{ animationDelay: `${d}ms` }} />
                ))}
              </span>
            </div>
          ) : (
            <div
              className={cn(
                "max-w-[92%] animate-in rounded-2xl rounded-bl-sm px-4 py-3 leading-6 fade-in-0 duration-300",
                abstained ? "border border-dashed text-muted-foreground" : "bg-muted",
              )}
            >
              {abstained && <CircleSlashIcon className="mr-1.5 mb-0.5 inline size-3.5" aria-hidden />}
              {item.answer.slice(0, typed)}
              {phase === "typing" && <span className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-blink bg-primary" />}
              {phase === "done" && !abstained && (
                <>
                  {" "}
                  <span className="rounded bg-primary/15 px-1 font-medium text-primary">[1]</span>.
                  <span className="mt-2.5 flex w-fit animate-in items-center gap-1.5 rounded-md border bg-background px-2 py-1 text-xs text-muted-foreground fade-in-0 slide-in-from-left-1 duration-500">
                    <FileTextIcon className="size-3" aria-hidden />
                    <span className="font-medium text-primary">[1]</span> {item.cite}
                  </span>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
