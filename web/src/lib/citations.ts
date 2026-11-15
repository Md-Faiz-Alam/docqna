/** Helpers for showing answers with clickable [n] citations. */

export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; id: number };

/** Split "Meals are capped [1][2]." into text and citation parts. */
export function splitCitations(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  const pattern = /\[(\d{1,2})\]/g;
  let last = 0;
  for (const match of answer.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: "text", text: answer.slice(last, index) });
    parts.push({ kind: "cite", id: Number(match[1]) });
    last = index + match[0].length;
  }
  if (last < answer.length) parts.push({ kind: "text", text: answer.slice(last) });
  return parts;
}

export function splitSentences(text: string): string[] {
  return (text.match(/[^.!?\n]+(?:[.!?]+["')\]]*|\n|$)/g) ?? [text])
    .map((s) => s.trim())
    .filter(Boolean);
}

const STOP_WORDS = new Set(
  "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split(
    " ",
  ),
);

function words(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+(?:[.,][0-9]+)?/g) ?? []).filter(
    (w) => !STOP_WORDS.has(w),
  );
}

/**
 * Index of the source sentence that shares the most words with the answer
 * (numbers count double), or -1 when nothing overlaps. This is a lexical
 * match for highlighting, not the backend's grounding score.
 */
export function closestSentence(answer: string, sentences: string[]): number {
  const answerWords = new Set(words(answer.replace(/\[\d+\]/g, "")));
  let best = -1;
  let bestScore = 0;
  sentences.forEach((sentence, index) => {
    let score = 0;
    for (const word of new Set(words(sentence))) {
      if (answerWords.has(word)) score += /\d/.test(word) ? 2 : 1;
    }
    if (score > bestScore) {
      bestScore = score;
      best = index;
    }
  });
  return best;
}

export function pageLabel(start: number | null, end: number | null): string | null {
  if (start === null) return null;
  return end === null || end === start ? `page ${start}` : `pages ${start}–${end}`;
}
