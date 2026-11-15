import type { Metadata } from "next";

import { PageHeader } from "@/components/page-header";
import { PercentBarChart, ThresholdChart } from "@/components/results/charts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { inventedAnswers, quality, results } from "@/lib/results";
import { GITHUB_URL } from "@/lib/site";
import { percent } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Results",
  description: "How DocQ&A was measured: retrieval, answer quality and made-up answers, with every number from a committed experiment.",
};

function Section({
  title,
  description,
  children,
}: {
  title: string;
  description: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">{title}</CardTitle>
        <CardDescription className="max-w-3xl text-pretty">{description}</CardDescription>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export default function ResultsPage() {
  const local = quality("ollama");
  const gemini = quality("gemini");
  const bench = results.public_benchmark;
  const totalUnanswerable = local.public.unanswerable + local.private.unanswerable;

  const answerData = [
    { label: "Before (phi3:mini)", public: results.old_gate.public_correct },
    { label: "After (phi3:mini)", public: local.public.correct, private: local.private.correct },
    { label: `After (Gemini)`, public: gemini.public.correct, private: gemini.private.correct },
  ];

  const rows = [
    {
      name: "phi3:mini, before recalibration",
      publicCorrect: results.old_gate.public_correct,
      publicFalse: results.old_gate.public_false_answers,
      privateCorrect: null,
      invented: null,
    },
    ...[local, gemini].map((q) => ({
      name: `${q.label}, after`,
      publicCorrect: q.public.correct,
      publicFalse: q.public.false_answers,
      privateCorrect: q.private.correct,
      invented: inventedAnswers(q),
    })),
  ];

  return (
    <>
    <PageHeader eyebrow="Evaluation" title="Measured results">
      Every number here comes from experiments committed to the{" "}
      <a href={GITHUB_URL} className="text-primary underline-offset-4 hover:underline" target="_blank" rel="noreferrer">
        repository
      </a>
      . Two test sets were used: a public one of {bench.answerable + bench.unanswerable} questions about four
      fictional company documents ({bench.unanswerable} of them deliberately unanswerable), and a private one of{" "}
      {local.private.cases} questions about a real document that is not published.
    </PageHeader>
    <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10 sm:px-6">

      <Section
        title="Correct answers, and no made-up ones"
        description={
          <>
            Share of answerable questions answered correctly. An answer counts as correct when it states the
            labelled fact or cites a passage that contains it. Across all {totalUnanswerable} unanswerable
            questions, neither model invented an answer after recalibration.
          </>
        }
      >
        <PercentBarChart
          data={answerData}
          series={[
            { key: "public", label: "Public test", color: "var(--color-chart-1)" },
            { key: "private", label: "Private test", color: "var(--color-chart-2)" },
          ]}
          description={`Public test correct answers: before ${percent(results.old_gate.public_correct)}, phi3:mini after ${percent(local.public.correct)}, Gemini after ${percent(gemini.public.correct)}.`}
        />
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[36rem] text-sm">
            <caption className="sr-only">Answer quality by configuration</caption>
            <thead className="text-left text-muted-foreground">
              <tr className="border-b">
                <th scope="col" className="py-2 pr-4 font-medium">Configuration</th>
                <th scope="col" className="py-2 pr-4 font-medium">Public: correct</th>
                <th scope="col" className="py-2 pr-4 font-medium">Public: made-up answers</th>
                <th scope="col" className="py-2 pr-4 font-medium">Private: correct</th>
                <th scope="col" className="py-2 font-medium">Made-up answers, both sets</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.name} className="border-b last:border-0">
                  <th scope="row" className="py-2 pr-4 text-left font-normal">{row.name}</th>
                  <td className="py-2 pr-4 tabular-nums">{percent(row.publicCorrect)}</td>
                  <td className="py-2 pr-4 tabular-nums">{percent(row.publicFalse)}</td>
                  <td className="py-2 pr-4 tabular-nums">{percent(row.privateCorrect)}</td>
                  <td className="py-2 tabular-nums">
                    {row.invented ? `${row.invented.count} of ${row.invented.total}` : "–"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-sm text-muted-foreground">
          Typical Gemini answer time: {gemini.public.median_seconds} s (public), {gemini.private.median_seconds} s
          (private). Thresholds were chosen and reported on the same questions, so these numbers are optimistic,
          and “0 made-up answers” means none observed in {totalUnanswerable}.
        </p>
      </Section>

      <Section
        title="Why one threshold was not enough"
        description="With the local model, a single retrieval threshold trades correct answers for made-up ones: loosening it answers more questions but also invents answers. The fix was a second check after writing: each answer is compared with the sentences it cites, and unsupported answers are withheld. The dashed line marks the gate that is now used together with that check."
      >
        <ThresholdChart
          data={results.gate_only_sweep_phi3_public}
          selected={-10.5}
          description="As the retrieval gate is loosened, correct answers rise, but so do made-up answers, which pass 60% at the loosest setting."
        />
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section
          title="Search method"
          description="How often the right passage ranks first (Hit@1), and mean reciprocal rank, on the private test. Combining keyword and meaning search and re-ranking with a second model worked best."
        >
          <PercentBarChart
            data={results.retrieval.map((r) => ({ label: r.label, hit_at_1: r.hit_at_1, mrr: r.mrr }))}
            series={[
              { key: "hit_at_1", label: "Hit@1", color: "var(--color-chart-1)" },
              { key: "mrr", label: "MRR", color: "var(--color-chart-3)" },
            ]}
            height={260}
            description="Hit@1 by search method: hybrid with reranker is highest."
          />
        </Section>
        <Section
          title="How documents are split"
          description="Simple fixed-size passages matched or beat the more complex splitting strategies, so the simplest one is used."
        >
          <PercentBarChart
            data={results.chunking.map((r) => ({ label: r.label, hit_at_1: r.hit_at_1, mrr: r.mrr }))}
            series={[
              { key: "hit_at_1", label: "Hit@1", color: "var(--color-chart-1)" },
              { key: "mrr", label: "MRR", color: "var(--color-chart-3)" },
            ]}
            height={260}
            description="Fixed-size chunking has the highest Hit@1."
          />
        </Section>
      </div>

      <Section
        title="Public benchmark: finding the evidence"
        description="Checked automatically on every code change: the build fails if search quality drops."
      >
        <PercentBarChart
          data={[
            { label: "Top 1", value: bench.hit_at_1 },
            { label: "Top 3", value: bench.hit_at_3 },
            { label: "Top 5", value: bench.hit_at_5 },
          ]}
          series={[{ key: "value", label: "Right passage found", color: "var(--color-chart-5)" }]}
          height={220}
          description={`The right passage is in the top 1 for ${percent(bench.hit_at_1)}, top 3 for ${percent(bench.hit_at_3)}, and top 5 for ${percent(bench.hit_at_5)} of answerable questions.`}
        />
      </Section>
    </div>
    </>
  );
}
