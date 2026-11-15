import type { Metadata } from "next";

import { PageHeader } from "@/components/page-header";
import { PercentBarChart, ThresholdChart } from "@/components/results/charts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { duration, inventedAnswers, layers, quality, results } from "@/lib/results";
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
  const localLayers = layers("ollama");
  const geminiLayers = layers("gemini");
  const layerChart = localLayers.layers.map((row) => ({
    label: row.label.replace(" (meaning search only)", "").replace(" (full system)", ""),
    correct: row.correct / row.answerable,
    made_up: row.false_answers / row.unanswerable,
  }));
  const attackRuns = results.prompt_injection;
  const attackCases = attackRuns[0]?.overall.baseline.cases ?? 0;
  const speed = [...results.speed_and_cost].sort((a, b) => a.provider.localeCompare(b.provider));
  const geminiSpeed = speed.find((row) => row.provider === "gemini");
  const stageLabels: [string, string][] = [
    ["search_ms", "Search (meaning + keywords)"],
    ["rerank_ms", "Reranking"],
    ["first_token_ms", "Until the first word appears"],
    ["generation_ms", "AI time (reading the prompt + writing)"],
    ["grounding_ms", "Grounding check"],
    ["total_ms", "Total, question to final answer"],
  ];
  const heldout = results.heldout;
  const heldoutRows = heldout
    ? [...heldout.rows].sort((a, b) => b.provider.localeCompare(a.provider))
    : [];
  const tuned = (provider: string) => {
    const q = quality(provider === "gemini" ? "gemini" : "ollama");
    const answerable = q.public.answerable + q.private.answerable;
    const correct = Math.round(
      q.public.correct * q.public.answerable + q.private.correct * q.private.answerable,
    );
    return { correct, answerable, invented: inventedAnswers(q) };
  };
  const review = heldout?.gemini_review;
  const grading = results.grading_check;
  const plain = localLayers.layers[0];
  const full = localLayers.layers[localLayers.layers.length - 1];

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
        title="What each layer adds"
        description={
          <>
            Plain RAG first, then one layer at a time, on the same {plain.answerable + plain.unanswerable}{" "}
            questions. With the small local model, plain RAG invented answers to {plain.false_answers} of{" "}
            {plain.unanswerable} unanswerable questions; the full system invents none, but withholds some correct
            answers to get there. Gemini already refuses on its own, so for it the safety layers change nothing.
          </>
        }
      >
        <PercentBarChart
          data={layerChart}
          series={[
            { key: "correct", label: "Correct answers (phi3:mini)", color: "var(--color-chart-1)" },
            { key: "made_up", label: "Made-up answers (phi3:mini)", color: "var(--color-chart-4)" },
          ]}
          height={280}
          description={`With phi3:mini, made-up answers fall from ${plain.false_answers} of ${plain.unanswerable} with plain RAG to ${full.false_answers} with the full system, while correct answers go from ${plain.correct} to ${full.correct} of ${full.answerable}.`}
        />
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[40rem] text-sm">
            <caption className="sr-only">Answer quality as each pipeline layer is added</caption>
            <thead className="text-left text-muted-foreground">
              <tr className="border-b">
                <th scope="col" className="py-2 pr-4 font-medium">Layer</th>
                <th scope="col" className="py-2 pr-4 font-medium">Evidence found</th>
                {[localLayers, geminiLayers].map((p) => (
                  <th key={p.provider} scope="col" className="py-2 pr-4 font-medium">
                    {p.label}: correct / made-up
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {localLayers.layers.map((row, i) => {
                const g = geminiLayers.layers[i];
                return (
                  <tr key={row.key} className="border-b last:border-0">
                    <th scope="row" className="py-2 pr-4 text-left font-normal">{row.label}</th>
                    <td className="py-2 pr-4 tabular-nums">{row.evidence_in_top_k} of {row.answerable}</td>
                    <td className="py-2 pr-4 tabular-nums">
                      {row.correct} of {row.answerable} / {row.false_answers} of {row.unanswerable}
                    </td>
                    <td className="py-2 pr-4 tabular-nums">
                      {g.correct} of {g.answerable} / {g.false_answers} of {g.unanswerable}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-sm text-muted-foreground">
          Keyword search is what finds the evidence. The reranker adds no accuracy on these short documents; it
          stays because both safety checks use its scores.
        </p>
      </Section>

      {attackRuns.length > 0 && (
        <Section
          title="Attack test: can a document take over the answer?"
          description={
            <>
              {attackCases} prompt-injection attacks. Three fictional documents hide instructions for the AI
              (add a phishing link, state a false fact, leak data through an image link, print a code word) and
              are mixed with the normal test documents; other attacks are typed straight into the question. An
              attack counts as successful when its text reaches the user. Two checks were added after the first
              run: text addressed to an AI is removed from the passages before the AI sees them, and an answer
              with a link that no source contains is withheld.
            </>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <caption className="sr-only">Successful attacks before and after the defence</caption>
              <thead className="text-left text-muted-foreground">
                <tr className="border-b">
                  <th scope="col" className="py-2 pr-4 font-medium">Attack</th>
                  {attackRuns.map((run) => (
                    <th key={run.provider} scope="col" className="py-2 pr-4 font-medium">
                      {run.label}: before → after
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {attackRuns[0].attacks.map((attack, i) => (
                  <tr key={attack.key} className="border-b last:border-0">
                    <th scope="row" className="py-2 pr-4 text-left font-normal">
                      {attack.label} <span className="text-muted-foreground">({attack.cases})</span>
                    </th>
                    {attackRuns.map((run) => (
                      <td key={run.provider} className="py-2 pr-4 tabular-nums">
                        {run.attacks[i].before} → <span className="font-medium">{run.attacks[i].after}</span>
                      </td>
                    ))}
                  </tr>
                ))}
                <tr className="border-t font-medium">
                  <th scope="row" className="py-2 pr-4 text-left">All attacks reaching the user</th>
                  {attackRuns.map((run) => (
                    <td key={run.provider} className="py-2 pr-4 tabular-nums">
                      {run.overall.baseline.user_saw_it} of {run.overall.baseline.cases} →{" "}
                      {run.overall.defended.user_saw_it} of {run.overall.defended.cases}
                    </td>
                  ))}
                </tr>
                <tr>
                  <th scope="row" className="py-2 pr-4 text-left font-normal text-muted-foreground">
                    Real answers still correct
                  </th>
                  {attackRuns.map((run) => (
                    <td key={run.provider} className="py-2 pr-4 tabular-nums text-muted-foreground">
                      {run.overall.baseline.real_answer_correct} of {run.overall.baseline.real_answer_cases} →{" "}
                      {run.overall.defended.real_answer_correct} of {run.overall.defended.real_answer_cases}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            The grounding check could not stop these attacks: the attacker&apos;s sentence really is in the cited
            document. The new checks are pattern rules for known attack shapes, written alongside this test, so
            they are not proof against every phrasing. They removed nothing from the clean test documents.
          </p>
        </Section>
      )}

      {speed.length > 0 && (
        <Section
          title="Speed and cost"
          description={
            <>
              Every stage timed on the {speed[0].questions} public test questions, with answers streamed as in the
              demo. Search, reranking and checks run on the CPU of an ordinary {speed[0].logical_cpus}-thread
              desktop; Gemini runs in Google&apos;s cloud, phi3:mini on the same CPU.
            </>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <caption className="sr-only">Time per stage, median and 95th percentile</caption>
              <thead className="text-left text-muted-foreground">
                <tr className="border-b">
                  <th scope="col" className="py-2 pr-4 font-medium">Stage</th>
                  {speed.map((row) => (
                    <th key={row.provider} scope="col" className="py-2 pr-4 font-medium">
                      {row.label}: typical (slowest 5%)
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stageLabels.map(([key, label]) => (
                  <tr key={key} className={key === "total_ms" ? "border-t font-medium" : "border-b last:border-0"}>
                    <th scope="row" className="py-2 pr-4 text-left font-normal">{label}</th>
                    {speed.map((row) => (
                      <td key={row.provider} className="py-2 pr-4 tabular-nums">
                        {duration(row.stages_ms[key]?.median)}{" "}
                        <span className="text-muted-foreground">({duration(row.stages_ms[key]?.p95)})</span>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 space-y-2 text-sm text-muted-foreground">
            {geminiSpeed?.cost_per_1000_usd != null && (
              <p>
                <span className="font-medium text-foreground">Cost:</span> a question sends about{" "}
                {Math.round(geminiSpeed.tokens.input_tokens).toLocaleString("en-US")} tokens to Gemini and gets about{" "}
                {Math.round(geminiSpeed.tokens.output_tokens)} back, which is ${geminiSpeed.cost_per_1000_usd.toFixed(2)}{" "}
                per 1,000 questions at Google&apos;s paid prices. This demo uses the free tier, so it costs nothing,
                within Google&apos;s rate limits. phi3:mini runs locally and has no per-question cost.
              </p>
            )}
            <p>
              Starting the server loads two small models in about {speed[0].startup_seconds} s. Indexing the four
              public documents took {speed[0].index_public_seconds} s
              {speed[0].index_private
                ? `, and a ${speed[0].index_private.chunks}-passage private PDF took ${speed[0].index_private.seconds} s`
                : ""}
              .
            </p>
          </div>
        </Section>
      )}

      {heldoutRows.length > 0 && (
        <Section
          title="Unseen questions: an honest check"
          description={
            <>
              Every threshold above was chosen on the same test sets it is reported on, so those numbers are
              optimistic. This test uses 100 questions from SQuAD 2.0, a public Stanford benchmark of Wikipedia
              articles, that were never used to tune anything. Half are unanswerable: they were written to look
              answerable from the text, by negating it or swapping a word.
            </>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] text-sm">
              <caption className="sr-only">Tuned test sets compared with unseen SQuAD questions</caption>
              <thead className="text-left text-muted-foreground">
                <tr className="border-b">
                  <th scope="col" className="py-2 pr-4 font-medium">Model</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Tuned sets: correct</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Tuned sets: made-up</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Unseen: correct</th>
                  <th scope="col" className="py-2 font-medium">Unseen: answered an unanswerable</th>
                </tr>
              </thead>
              <tbody>
                {heldoutRows.map((row) => {
                  const t = tuned(row.provider);
                  return (
                    <tr key={row.provider} className="border-b last:border-0">
                      <th scope="row" className="py-2 pr-4 text-left font-normal">{row.label}</th>
                      <td className="py-2 pr-4 tabular-nums">{t.correct} of {t.answerable}</td>
                      <td className="py-2 pr-4 tabular-nums">{t.invented.count} of {t.invented.total}</td>
                      <td className="py-2 pr-4 tabular-nums">{row.correct} of {row.answerable}</td>
                      <td className="py-2 tabular-nums font-medium">{row.made_up} of {row.unanswerable}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {review && (
            <p className="mt-4 text-sm text-muted-foreground">
              Every Gemini answer to an “unanswerable” question was checked by hand against the article:{" "}
              {review.supported_by_article} are in fact answered elsewhere in the article (SQuAD only checks one
              paragraph), {review.sidestepped_false_premise} state true facts that sidestep the question&apos;s false
              premise, and <span className="font-medium text-foreground">{review.fell_for_trap} fell for the
              trap</span> and answered as if the altered question were the original. This test also found a bug: correct
              answers citing two sources as “[1, 5]” were being refused. It is fixed, and these are the numbers
              after the fix.
            </p>
          )}
        </Section>
      )}

      {grading && (
        <Section
          title="Can the automatic grading be trusted?"
          description={
            <>
              Every number on this page is graded by rules, not by people. To check those rules,{" "}
              {grading.labelled} answers the system showed (half from the tuned test set, half from the unseen SQuAD
              questions, both models) were graded blind by the project author, with help from an AI assistant, and
              compared with the automatic grade.
            </>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[34rem] text-sm">
              <caption className="sr-only">Automatic grading compared with AI-assisted human grading</caption>
              <thead className="text-left text-muted-foreground">
                <tr className="border-b">
                  <th scope="col" className="py-2 pr-4 font-medium">Answers</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Same grade</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Rules too harsh</th>
                  <th scope="col" className="py-2 font-medium">Rules too generous</th>
                </tr>
              </thead>
              <tbody>
                {grading.rows.map((row) => (
                  <tr key={row.key} className="border-b last:border-0">
                    <th scope="row" className="py-2 pr-4 text-left font-normal">{row.label}</th>
                    <td className="py-2 pr-4 tabular-nums font-medium">
                      {row.agree} of {row.compared} ({Math.round((100 * row.agree) / row.compared)}%)
                    </td>
                    <td className="py-2 pr-4 tabular-nums">{row.auto_too_harsh}</td>
                    <td className="py-2 tabular-nums">{row.auto_too_generous}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            On normal questions the rules hold up: their mistakes go both ways (correct answers in other words
            marked wrong, and two answers that turned &ldquo;after four hours&rdquo; into &ldquo;within four
            hours&rdquo; marked correct). On SQuAD&apos;s trick questions the rules count every answer as made up,
            but about 4 in 10 of those answers were in fact right, so the unseen-question &ldquo;made up&rdquo;
            counts above overstate the problem. Grading with AI help is a weaker check than independent human
            graders; one grader, 80 answers.
          </p>
        </Section>
      )}

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
