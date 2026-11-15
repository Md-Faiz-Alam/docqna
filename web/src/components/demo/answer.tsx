"use client";

import { CircleSlashIcon, ThumbsDownIcon, ThumbsUpIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { splitCitations } from "@/lib/citations";
import type { Citation, QAResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

export type Rating = "up" | "down";

/** Answer text with each [n] rendered as a button that opens its source. */
export function AnswerText({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: Citation[];
  onCite: (citation: Citation) => void;
}) {
  return (
    <p className="whitespace-pre-wrap leading-7">
      {splitCitations(text).map((part, index) => {
        if (part.kind === "text") return <span key={index}>{part.text}</span>;
        const source = sources.find((s) => s.citation_id === part.id);
        if (!source) return <span key={index}>[{part.id}]</span>;
        return (
          <button
            key={index}
            type="button"
            onClick={() => onCite(source)}
            className="mx-0.5 rounded bg-primary/15 px-1 align-baseline text-sm font-medium text-primary transition-colors hover:bg-primary/25 focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
            aria-label={`Open source ${part.id}: ${source.filename}`}
          >
            [{part.id}]
          </button>
        );
      })}
    </p>
  );
}

export function AnswerCard({
  response,
  rating,
  onCite,
  onRate,
}: {
  response: QAResponse;
  rating?: Rating;
  onCite: (citation: Citation) => void;
  onRate: (rating: Rating) => void;
}) {
  const cited = response.sources.filter((s) => response.cited_source_ids.includes(s.citation_id));

  return (
    <div className="flex flex-col gap-3">
      {response.abstained ? (
        <p className="flex items-start gap-2 text-muted-foreground">
          <CircleSlashIcon className="mt-1 size-4 shrink-0" aria-hidden />
          <span>{response.answer}</span>
        </p>
      ) : (
        <AnswerText text={response.answer} sources={response.sources} onCite={onCite} />
      )}

      {cited.length > 0 && (
        <ul className="flex flex-wrap gap-1.5" aria-label="Sources">
          {cited.map((source) => (
            <li key={source.citation_id}>
              <button
                type="button"
                onClick={() => onCite(source)}
                className="max-w-64 truncate rounded-md border bg-background px-2 py-1 text-left text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                <span className="font-medium text-primary">[{source.citation_id}]</span>{" "}
                {source.filename}
                {source.page_start !== null && ` · p. ${source.page_start}`}
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        {response.standalone_question && (
          <span className="mr-auto truncate" title={response.standalone_question}>
            Searched for: “{response.standalone_question}”
          </span>
        )}
        <span className={cn(!response.standalone_question && "ml-auto")}>
          {rating ? "Thanks for the feedback" : "Was this helpful?"}
        </span>
        {(["up", "down"] as const).map((value) => {
          const Icon = value === "up" ? ThumbsUpIcon : ThumbsDownIcon;
          return (
            <Button
              key={value}
              variant="ghost"
              size="icon-sm"
              aria-label={value === "up" ? "Helpful answer" : "Unhelpful answer"}
              aria-pressed={rating === value}
              disabled={rating !== undefined}
              onClick={() => onRate(value)}
              className={cn(rating === value && "text-primary")}
            >
              <Icon />
            </Button>
          );
        })}
      </div>
    </div>
  );
}
