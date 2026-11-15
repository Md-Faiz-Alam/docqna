"use client";

import { FileTextIcon } from "lucide-react";
import { useMemo } from "react";

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { closestSentence, pageLabel, splitSentences } from "@/lib/citations";
import type { Citation } from "@/lib/types";

export function CitationSheet({
  citation,
  answer,
  onClose,
}: {
  citation: Citation | null;
  answer: string;
  onClose: () => void;
}) {
  const sentences = useMemo(() => (citation ? splitSentences(citation.text) : []), [citation]);
  const highlighted = useMemo(() => closestSentence(answer, sentences), [answer, sentences]);

  const location = citation
    ? [pageLabel(citation.page_start, citation.page_end), citation.section]
        .filter(Boolean)
        .join(" · ")
    : "";

  return (
    <Sheet open={citation !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent aria-describedby="citation-description">
        {citation && (
          <>
            <SheetHeader>
              <SheetTitle className="flex items-center gap-2">
                <span className="rounded bg-primary/15 px-1.5 text-primary">[{citation.citation_id}]</span>
                Source passage
              </SheetTitle>
              <SheetDescription id="citation-description" className="flex items-center gap-1.5">
                <FileTextIcon className="size-3.5" aria-hidden />
                <span className="break-all">{citation.filename}</span>
                {location && <span>· {location}</span>}
              </SheetDescription>
            </SheetHeader>
            <div className="flex-1 overflow-y-auto px-5 pb-6">
              <p className="text-sm leading-7">
                {sentences.map((sentence, index) =>
                  index === highlighted ? (
                    <mark key={index} className="rounded bg-highlight px-0.5 text-foreground">
                      {sentence}{" "}
                    </mark>
                  ) : (
                    <span key={index}>{sentence} </span>
                  ),
                )}
              </p>
              {highlighted >= 0 && (
                <p className="mt-4 text-xs text-muted-foreground">
                  Highlighted: the sentence in this passage that best matches the answer’s wording.
                </p>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
