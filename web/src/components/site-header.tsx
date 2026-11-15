"use client";

import { ArrowRightIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { Logo } from "@/components/logo";
import { ThemePicker } from "@/components/theme-picker";
import { Button } from "@/components/ui/button";
import { GITHUB_URL } from "@/lib/site";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/demo", label: "Demo" },
  { href: "/results", label: "Results" },
];

function GitHubIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden {...props}>
      <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.29-1.69-1.29-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.71 1.26 3.37.96.1-.75.4-1.26.73-1.55-2.56-.29-5.24-1.28-5.24-5.7 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.06 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.77.11 3.06.74.81 1.19 1.84 1.19 3.1 0 4.43-2.69 5.41-5.25 5.69.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
    </svg>
  );
}

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="glass sticky top-0 z-40 border-b border-border/60">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
      >
        Skip to content
      </a>
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-2 px-4 sm:gap-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight" aria-label="DocQ&A home">
          <Logo className="size-7" />
          <span className="hidden sm:inline">
            DocQ<span className="text-primary">&amp;</span>A
          </span>
        </Link>

        <nav aria-label="Main" className="flex items-center rounded-full sm:ml-2 border border-border/70 bg-muted/50 p-1 text-sm">
          {LINKS.map((link) => {
            const active = link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "rounded-full px-3 py-1 text-muted-foreground transition-all duration-200 hover:text-foreground",
                  active && "bg-background text-foreground shadow-sm",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-1">
          <Button variant="ghost" size="icon" asChild className="hidden sm:inline-flex">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub">
              <GitHubIcon className="size-[18px]" />
            </a>
          </Button>
          <ThemePicker />
          {pathname !== "/demo" && (
            <Button size="sm" asChild className="ml-1 hidden rounded-full md:inline-flex">
              <Link href="/demo">
                Try it <ArrowRightIcon />
              </Link>
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
