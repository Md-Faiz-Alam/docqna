/** Colour themes. Each sets the accent hue in globals.css ([data-accent]). */

export const ACCENTS = [
  { id: "indigo", label: "Indigo", swatch: "oklch(0.55 0.2 268)" },
  { id: "violet", label: "Violet", swatch: "oklch(0.55 0.21 300)" },
  { id: "ocean", label: "Ocean", swatch: "oklch(0.56 0.17 235)" },
  { id: "teal", label: "Teal", swatch: "oklch(0.58 0.12 185)" },
  { id: "emerald", label: "Emerald", swatch: "oklch(0.58 0.15 155)" },
  { id: "sunset", label: "Sunset", swatch: "oklch(0.64 0.18 45)" },
  { id: "rose", label: "Rose", swatch: "oklch(0.6 0.2 10)" },
  { id: "mono", label: "Mono", swatch: "oklch(0.25 0 0)" },
] as const;

export type AccentId = (typeof ACCENTS)[number]["id"];

export const DEFAULT_ACCENT: AccentId = "indigo";
export const ACCENT_STORAGE_KEY = "docqna_accent";

export function isAccent(value: unknown): value is AccentId {
  return ACCENTS.some((a) => a.id === value);
}

/**
 * Runs in <head> before the page paints, so a saved theme never flickers.
 * Kept tiny and self-contained (it is inlined as a string).
 */
export const ACCENT_INIT_SCRIPT = `(function(){try{var a=localStorage.getItem(${JSON.stringify(
  ACCENT_STORAGE_KEY,
)});var ok=${JSON.stringify(ACCENTS.map((x) => x.id))};document.documentElement.setAttribute("data-accent",ok.indexOf(a)>=0?a:${JSON.stringify(
  DEFAULT_ACCENT,
)});}catch(e){document.documentElement.setAttribute("data-accent",${JSON.stringify(DEFAULT_ACCENT)});}})();`;
