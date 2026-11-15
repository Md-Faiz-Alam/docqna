"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Counts from 0 to `value` once it is on screen. The final text is always
 * in the page (screen readers and no-JS see the real number).
 */
export function CountUp({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  durationMs = 1400,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  durationMs?: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const [shown, setShown] = useState<number | null>(null);
  const format = (n: number) => `${prefix}${n.toFixed(decimals).replace(/\.0+$/, "")}${suffix}`;

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof IntersectionObserver === "undefined") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;

    let frame = 0;
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((e) => e.isIntersecting)) return;
        observer.disconnect();
        const start = performance.now();
        const tick = (now: number) => {
          const t = Math.min(1, (now - start) / durationMs);
          const eased = 1 - Math.pow(1 - t, 3);
          setShown(value * eased);
          if (t < 1) frame = requestAnimationFrame(tick);
          else setShown(null);
        };
        setShown(0);
        frame = requestAnimationFrame(tick);
      },
      { threshold: 0.4 },
    );
    observer.observe(node);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [value, durationMs]);

  return (
    <span ref={ref} className="tabular-nums">
      <span aria-hidden={shown !== null}>{shown === null ? format(value) : format(shown)}</span>
      {shown !== null && <span className="sr-only">{format(value)}</span>}
    </span>
  );
}
