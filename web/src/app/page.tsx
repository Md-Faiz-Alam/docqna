import {
  ArrowRightIcon,
  BookOpenCheckIcon,
  CircleSlashIcon,
  FileUpIcon,
  HardDriveIcon,
  LayersIcon,
  MessagesSquareIcon,
  QuoteIcon,
  SearchIcon,
  ShieldCheckIcon,
  SparklesIcon,
  ZapIcon,
} from "lucide-react";
import Link from "next/link";

import { CountUp } from "@/components/count-up";
import { LiveChat } from "@/components/landing/live-chat";
import { Reveal } from "@/components/reveal";
import { Button } from "@/components/ui/button";
import { inventedAnswers, quality, results } from "@/lib/results";
import { cn, percent } from "@/lib/utils";

const STEPS = [
  { icon: FileUpIcon, title: "Upload", text: "PDF, Word or text. Scanned PDFs are read with OCR." },
  { icon: LayersIcon, title: "Index", text: "Split into short passages, indexed by meaning and by keywords." },
  { icon: SearchIcon, title: "Retrieve", text: "Both searches are fused, then a second model re-ranks the best passages." },
  { icon: ShieldCheckIcon, title: "Gate", text: "If nothing relevant is found, it says so instead of guessing." },
  { icon: SparklesIcon, title: "Answer", text: "A short answer written only from those passages, each claim cited [1]." },
  { icon: BookOpenCheckIcon, title: "Verify", text: "The answer is checked against the cited sentences, or withheld." },
];

function Eyebrow({ children }: { children: React.ReactNode }) {
  return <p className="text-sm font-semibold tracking-wide text-primary uppercase">{children}</p>;
}

function BentoCard({
  className,
  icon: Icon,
  title,
  text,
  children,
  delay = 0,
}: {
  className?: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  text: string;
  children?: React.ReactNode;
  delay?: number;
}) {
  return (
    <Reveal delay={delay} className={cn("card-glow flex flex-col gap-4 overflow-hidden rounded-2xl border bg-card p-6", className)}>
      <span className="flex size-10 items-center justify-center rounded-xl bg-accent text-accent-foreground">
        <Icon className="size-5" aria-hidden />
      </span>
      <div>
        <h3 className="font-semibold tracking-tight">{title}</h3>
        <p className="mt-1.5 text-sm text-muted-foreground text-pretty">{text}</p>
      </div>
      {children}
    </Reveal>
  );
}

export default function HomePage() {
  const gemini = quality("gemini");
  const local = quality("ollama");
  const invented = inventedAnswers(gemini);
  const totalQuestions = results.public_benchmark.answerable + results.public_benchmark.unanswerable;

  const comparison = [
    { label: "Before tuning", value: results.old_gate.public_correct },
    { label: "Local model", value: local.public.correct },
    { label: "Gemini", value: gemini.public.correct },
  ];

  return (
    <>
      {/* ---------------------------------------------------------- hero */}
      <section className="relative isolate overflow-hidden">
        <div aria-hidden className="absolute inset-0 -z-10 overflow-hidden">
          <div className="bg-grid absolute inset-0" />
          <div className="absolute -top-40 left-[8%] size-[34rem] animate-aurora rounded-full bg-glow-1 opacity-30 blur-[110px] dark:opacity-40" />
          <div className="absolute -top-24 right-[4%] size-[28rem] animate-aurora rounded-full bg-glow-2 opacity-25 blur-[110px] [animation-delay:-6s] dark:opacity-35" />
          <div className="absolute top-60 left-1/3 size-[22rem] animate-aurora rounded-full bg-glow-3 opacity-20 blur-[100px] [animation-delay:-12s]" />
        </div>

        <div className="mx-auto grid max-w-6xl items-center gap-14 px-4 pt-16 pb-20 sm:px-6 md:pt-24 lg:grid-cols-[1.05fr_1fr] lg:pb-28">
          <div className="flex flex-col gap-7">
            <Link
              href="/results"
              className="group flex w-fit animate-in items-center gap-2 rounded-full border bg-card/70 py-1 pr-3 pl-1 text-xs shadow-sm backdrop-blur fade-in-0 slide-in-from-bottom-2 duration-700"
            >
              <span className="rounded-full bg-primary px-2 py-0.5 font-medium text-primary-foreground">New</span>
              <span className="text-muted-foreground">
                {percent(gemini.public.correct)} correct, 0 made-up answers
              </span>
              <ArrowRightIcon className="size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5" aria-hidden />
            </Link>

            <h1 className="animate-in text-4xl font-semibold tracking-tight text-balance fade-in-0 slide-in-from-bottom-3 duration-700 sm:text-6xl">
              Ask your documents.{" "}
              <span className="text-gradient">Check every answer.</span>
            </h1>

            <p className="max-w-xl animate-in text-lg text-muted-foreground text-pretty fade-in-0 slide-in-from-bottom-3 duration-1000">
              DocQ&amp;A answers questions about your PDF, Word and text files. Every answer cites the
              exact passage it came from, and when your documents don’t contain the answer, it says so
              instead of making one up.
            </p>

            <div className="flex animate-in flex-wrap gap-3 fade-in-0 slide-in-from-bottom-4 duration-1000">
              <Button size="lg" asChild className="group rounded-full shadow-lg shadow-primary/25">
                <Link href="/demo">
                  Try the live demo
                  <ArrowRightIcon className="transition-transform group-hover:translate-x-0.5" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild className="rounded-full bg-card/60 backdrop-blur">
                <Link href="/results">See how it’s measured</Link>
              </Button>
            </div>

            <ul className="flex flex-wrap gap-x-5 gap-y-2 text-sm text-muted-foreground">
              {["No sign-up", "Sample documents included", "Open source"].map((item) => (
                <li key={item} className="flex items-center gap-1.5">
                  <span className="size-1.5 rounded-full bg-success" aria-hidden />
                  {item}
                </li>
              ))}
            </ul>
          </div>

          <div className="animate-in fade-in-0 zoom-in-95 duration-1000 lg:animate-float">
            <LiveChat />
          </div>
        </div>
      </section>

      {/* --------------------------------------------------------- stats */}
      <section aria-labelledby="numbers" className="border-y bg-muted/30">
        <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
          <h2 id="numbers" className="sr-only">
            Measured quality
          </h2>
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                value: <CountUp value={results.public_benchmark.hit_at_5 * 100} suffix="%" />,
                label: "Right passage found in the top 5",
              },
              {
                value: <CountUp value={gemini.public.correct * 100} decimals={1} suffix="%" />,
                label: `Answerable questions answered correctly`,
              },
              {
                value: (
                  <>
                    <CountUp value={invented.count} /> <span className="text-2xl text-muted-foreground">/ {invented.total}</span>
                  </>
                ),
                label: "Made-up answers to unanswerable questions",
              },
              {
                value: <CountUp value={gemini.public.median_seconds ?? 2} decimals={1} suffix=" s" />,
                label: "Typical time to an answer",
              },
            ].map((stat, i) => (
              <Reveal as="li" key={stat.label} delay={i * 90} className="rounded-2xl border bg-card p-6 text-center shadow-sm">
                <p className="text-4xl font-semibold tracking-tight text-primary">{stat.value}</p>
                <p className="mt-2 text-sm text-muted-foreground">{stat.label}</p>
              </Reveal>
            ))}
          </ul>
          <p className="mt-6 text-center text-sm text-muted-foreground">
            Measured on {totalQuestions} public questions about fictional company documents, plus a private
            set.{" "}
            <Link href="/results" className="font-medium text-primary underline-offset-4 hover:underline">
              Methodology
            </Link>
          </p>
        </div>
      </section>

      {/* --------------------------------------------------------- bento */}
      <section aria-labelledby="features" className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <Reveal className="max-w-2xl">
          <Eyebrow>Built for trust</Eyebrow>
          <h2 id="features" className="mt-2 text-3xl font-semibold tracking-tight text-balance sm:text-4xl">
            Answers you can verify in one click
          </h2>
          <p className="mt-3 text-muted-foreground text-pretty">
            Most chat-with-PDF tools sound confident even when they are wrong. DocQ&amp;A is designed and
            measured around one rule: never invent an answer.
          </p>
        </Reveal>

        <div className="mt-10 grid gap-4 md:grid-cols-6">
          <BentoCard
            className="md:col-span-4"
            icon={QuoteIcon}
            title="Every claim is cited"
            text="Click any [number] to open the exact source passage, with the matching sentence highlighted."
          >
            <div className="rounded-xl border bg-muted/40 p-4 text-sm leading-6">
              <span className="text-muted-foreground">…3. Limited Hardware Warranty. </span>
              <mark className="rounded bg-highlight px-1 text-foreground">
                The CloudBox S2 enclosure is covered by a limited hardware warranty of three years from the date
                of purchase.
              </mark>
              <span className="text-muted-foreground"> Drives supplied with the device are covered for two years…</span>
            </div>
          </BentoCard>

          <BentoCard
            className="md:col-span-2"
            icon={CircleSlashIcon}
            title="Honest when it can’t answer"
            text="Off-topic or unsupported questions get a clear “not enough information”."
            delay={80}
          >
            <p className="mt-auto rounded-xl border border-dashed p-3 text-xs text-muted-foreground">
              The provided documents do not contain enough information to answer the question.
            </p>
          </BentoCard>

          <BentoCard
            className="md:col-span-2"
            icon={SearchIcon}
            title="Hybrid search + reranking"
            text="Keyword and meaning search are fused, then a cross-encoder picks the best passages."
          />

          <BentoCard
            className="md:col-span-2"
            icon={MessagesSquareIcon}
            title="Natural follow-ups"
            text="“And for flights?” is rewritten into a full question before searching."
            delay={80}
          />

          <BentoCard
            className="md:col-span-2"
            icon={ZapIcon}
            title="Streams in seconds"
            text="Answers appear word by word, then are checked before they are final."
            delay={160}
          />

          <BentoCard
            className="md:col-span-3"
            icon={BookOpenCheckIcon}
            title="Measured, not promised"
            text="Correct answers on the public test, before and after tuning the safety checks."
          >
            <div className="flex flex-col gap-2.5">
              {comparison.map((row) => (
                <div key={row.label} className="grid grid-cols-[6.5rem_1fr_3.5rem] items-center gap-3 text-xs">
                  <span className="text-muted-foreground">{row.label}</span>
                  <span className="h-2 overflow-hidden rounded-full bg-muted">
                    <span
                      className="block h-full rounded-full bg-gradient-to-r from-primary to-glow-2"
                      style={{ width: `${row.value * 100}%` }}
                    />
                  </span>
                  <span className="text-right font-medium tabular-nums">{percent(row.value)}</span>
                </div>
              ))}
            </div>
          </BentoCard>

          <BentoCard
            className="md:col-span-3"
            icon={HardDriveIcon}
            title="Private mode, fully offline"
            text="Run the same pipeline on your own computer with a local model. Nothing leaves your machine."
            delay={80}
          >
            <code className="mt-auto rounded-lg border bg-muted/50 px-3 py-2 font-mono text-xs text-muted-foreground">
              docker compose up -d
            </code>
          </BentoCard>
        </div>
      </section>

      {/* -------------------------------------------------- how it works */}
      <section aria-labelledby="how" className="border-y bg-muted/30">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <Reveal className="max-w-2xl">
            <Eyebrow>How it works</Eyebrow>
            <h2 id="how" className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
              An open-book exam, done carefully
            </h2>
            <p className="mt-3 text-muted-foreground">Find the right page first. Answer only from it. Then check.</p>
          </Reveal>

          <ol className="relative mt-12 grid gap-8 md:grid-cols-3 lg:grid-cols-6 lg:gap-4">
            <span
              aria-hidden
              className="absolute top-5 right-[8%] left-[8%] hidden h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent lg:block"
            />
            {STEPS.map((step, i) => (
              <Reveal as="li" key={step.title} delay={i * 80} className="relative flex flex-col gap-3">
                <span className="relative z-10 flex size-10 items-center justify-center rounded-full border bg-card text-primary shadow-sm">
                  <step.icon className="size-4.5" aria-hidden />
                </span>
                <div>
                  <h3 className="font-semibold">
                    <span className="mr-1 font-mono text-xs text-muted-foreground">{String(i + 1).padStart(2, "0")}</span>
                    {step.title}
                  </h3>
                  <p className="mt-1 text-sm text-muted-foreground text-pretty">{step.text}</p>
                </div>
              </Reveal>
            ))}
          </ol>
        </div>
      </section>

      {/* ------------------------------------------------------------ CTA */}
      <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <Reveal className="relative isolate overflow-hidden rounded-3xl border px-6 py-14 text-center shadow-xl sm:px-12">
          <div aria-hidden className="absolute inset-0 -z-10 bg-gradient-to-br from-primary/15 via-card to-glow-2/15" />
          <div aria-hidden className="absolute -top-24 left-1/2 -z-10 size-[30rem] -translate-x-1/2 animate-aurora rounded-full bg-glow-1 opacity-20 blur-[100px]" />
          <h2 className="text-3xl font-semibold tracking-tight text-balance sm:text-4xl">See it answer, and refuse, in real time</h2>
          <p className="mx-auto mt-3 max-w-xl text-muted-foreground">
            Load four sample documents with one click, or bring your own. This public demo uses Google
            Gemini, so please don’t upload anything private.
          </p>
          <Button size="lg" asChild className="group mt-8 rounded-full shadow-lg shadow-primary/25">
            <Link href="/demo">
              Open the demo <ArrowRightIcon className="transition-transform group-hover:translate-x-0.5" />
            </Link>
          </Button>
        </Reveal>
      </section>
    </>
  );
}
