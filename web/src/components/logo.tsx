import { cn } from "@/lib/utils";

/** DocQ&A mark: a document with a highlighted line, on an accent gradient. */
export function Logo({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-flex shrink-0 items-center justify-center rounded-[28%] shadow-sm", className)}
      style={{ background: "linear-gradient(135deg, var(--primary), var(--glow-2))" }}
    >
      <svg viewBox="0 0 32 32" className="size-full">
        <path d="M10 8h8l4 4v11a1 1 0 0 1-1 1H10a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" fill="white" opacity="0.95" />
        <rect x="11.5" y="15" width="8" height="2" rx="1" fill="var(--primary)" />
        <rect x="11.5" y="19" width="5.5" height="1.6" rx="0.8" fill="var(--primary)" opacity="0.45" />
      </svg>
    </span>
  );
}
