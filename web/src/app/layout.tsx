import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import { Logo } from "@/components/logo";
import { SiteHeader } from "@/components/site-header";
import { ThemeProvider } from "@/components/theme-provider";
import { ACCENT_INIT_SCRIPT, DEFAULT_ACCENT } from "@/lib/accents";
import { GITHUB_URL } from "@/lib/site";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "DocQ&A: answers from your documents, with sources", template: "%s · DocQ&A" },
  description:
    "Ask questions about your PDF, Word or text documents and get answers that cite the exact passage, or an honest “not enough information”.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fcfcfe" },
    { media: "(prefers-color-scheme: dark)", color: "#121218" },
  ],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      data-accent={DEFAULT_ACCENT}
      data-scroll-behavior="smooth"
      suppressHydrationWarning
    >
      <head>
        {/* Applies the saved colour theme before the first paint. */}
        <script dangerouslySetInnerHTML={{ __html: ACCENT_INIT_SCRIPT }} />
      </head>
      <body className={`${geistSans.variable} ${geistMono.variable} flex min-h-dvh flex-col font-sans`}>
        <ThemeProvider>
          <SiteHeader />
          <main id="main" className="flex-1">
            {children}
          </main>
          <footer className="border-t">
            <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 text-sm text-muted-foreground sm:flex-row sm:items-center sm:justify-between sm:px-6">
              <div className="flex items-center gap-2">
                <Logo className="size-5" />
                <span>DocQ&amp;A · answers you can check</span>
              </div>
              <nav aria-label="Footer" className="flex gap-5">
                <Link href="/demo" className="transition-colors hover:text-foreground">Demo</Link>
                <Link href="/results" className="transition-colors hover:text-foreground">Results</Link>
                <a href={GITHUB_URL} className="transition-colors hover:text-foreground" target="_blank" rel="noreferrer">
                  GitHub
                </a>
              </nav>
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
