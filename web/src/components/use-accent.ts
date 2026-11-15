"use client";

import { useSyncExternalStore } from "react";

import { ACCENT_STORAGE_KEY, type AccentId, DEFAULT_ACCENT, isAccent } from "@/lib/accents";

const listeners = new Set<() => void>();

function read(): AccentId {
  const value = document.documentElement.getAttribute("data-accent");
  return isAccent(value) ? value : DEFAULT_ACCENT;
}

export function setAccent(accent: AccentId) {
  document.documentElement.setAttribute("data-accent", accent);
  try {
    window.localStorage.setItem(ACCENT_STORAGE_KEY, accent);
  } catch {
    // Storage blocked: the theme still applies for this page.
  }
  listeners.forEach((listener) => listener());
}

/** The active colour theme (the default during server rendering). */
export function useAccent(): AccentId {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    read,
    () => DEFAULT_ACCENT,
  );
}
