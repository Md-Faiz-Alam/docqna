import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** 0.6949 -> "69.5%", 1 -> "100%", null -> "–" */
export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "–";
  const fixed = (value * 100).toFixed(digits);
  return `${fixed.replace(/\.0+$/, "")}%`;
}
