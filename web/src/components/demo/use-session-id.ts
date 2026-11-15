"use client";

import { useSyncExternalStore } from "react";

import { getSessionId, resetSessionId } from "@/lib/session";

let current: string | null = null;
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): string {
  current ??= getSessionId();
  return current;
}

/** The anonymous session ID: null during server rendering, stable in the browser. */
export function useSessionId(): [string | null, () => void] {
  const sessionId = useSyncExternalStore(subscribe, getSnapshot, () => null);
  const reset = () => {
    current = resetSessionId();
    listeners.forEach((listener) => listener());
  };
  return [sessionId, reset];
}
