"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { Info } from "@/lib/types";

export type ServerState = "checking" | "waking" | "ready" | "down";

/** How long a free server may take to wake up before we give up. */
const WAKE_TIMEOUT_MS = 3 * 60_000;
const RETRY_MS = 4_000;

/**
 * Wait for the backend. A free Hugging Face Space sleeps when idle and takes
 * a minute or two to wake, so a slow or failed first check shows "waking up"
 * and keeps retrying instead of failing straight away.
 */
export function useServer() {
  const [state, setState] = useState<ServerState>("checking");
  const [info, setInfo] = useState<Info | null>(null);
  const [attempt, setAttempt] = useState(0);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    const started = Date.now();
    const slow = setTimeout(() => {
      if (!cancelled.current) setState((s) => (s === "checking" ? "waking" : s));
    }, 2_500);

    async function poll(): Promise<void> {
      while (!cancelled.current) {
        try {
          await api.health(10_000);
          const loaded = await api.info();
          if (cancelled.current) return;
          setInfo(loaded);
          setState("ready");
          return;
        } catch {
          if (cancelled.current) return;
          if (Date.now() - started > WAKE_TIMEOUT_MS) {
            setState("down");
            return;
          }
          setState("waking");
          await new Promise((resolve) => setTimeout(resolve, RETRY_MS));
        }
      }
    }

    void poll();
    return () => {
      cancelled.current = true;
      clearTimeout(slow);
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setState("checking");
    setAttempt((n) => n + 1);
  }, []);

  return { state, info, retry };
}
