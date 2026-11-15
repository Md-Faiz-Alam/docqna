"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { percent } from "@/lib/utils";

const axis = { stroke: "var(--color-muted-foreground)", fontSize: 12, tickLine: false };
const tooltipStyle = {
  contentStyle: {
    background: "var(--color-popover)",
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    color: "var(--color-popover-foreground)",
    fontSize: 13,
  },
  cursor: { fill: "var(--color-muted)", opacity: 0.5 },
};
const asPercent = (v: number) => percent(v, 0);
const tooltipPercent = (v: unknown) => percent(Number(v));

export interface BarSeries {
  key: string;
  label: string;
  color: string;
}

export function PercentBarChart({
  data,
  series,
  categoryKey = "label",
  height = 280,
  description,
}: {
  data: Record<string, string | number | undefined>[];
  series: BarSeries[];
  categoryKey?: string;
  height?: number;
  description: string;
}) {
  return (
    <figure className="flex flex-col gap-2">
      <div style={{ height }} role="img" aria-label={description}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey={categoryKey} {...axis} interval={0} />
            <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} axisLine={false} />
            <Tooltip formatter={tooltipPercent} {...tooltipStyle} />
            {series.length > 1 && <Legend wrapperStyle={{ fontSize: 13 }} />}
            {series.map((s) => (
              <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[4, 4, 0, 0]} maxBarSize={56} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="sr-only">{description}</figcaption>
    </figure>
  );
}

export function ThresholdChart({
  data,
  selected,
  description,
}: {
  data: { threshold: number; correct: number; false_answers: number }[];
  selected: number;
  description: string;
}) {
  return (
    <figure className="flex flex-col gap-2">
      <div className="h-72" role="img" aria-label={description}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke="var(--color-border)" />
            <XAxis dataKey="threshold" type="number" domain={["dataMin", "dataMax"]} {...axis} />
            <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} axisLine={false} />
            <Tooltip
              formatter={tooltipPercent}
              labelFormatter={(v) => `Retrieval gate ≥ ${v}`}
              {...tooltipStyle}
            />
            <Legend wrapperStyle={{ fontSize: 13 }} />
            <ReferenceLine x={selected} stroke="var(--color-muted-foreground)" strokeDasharray="4 4" />
            <Line
              type="monotone"
              dataKey="correct"
              name="Answerable questions answered correctly"
              stroke="var(--color-chart-1)"
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="false_answers"
              name="Unanswerable questions given a made-up answer"
              stroke="var(--color-chart-4)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="sr-only">{description}</figcaption>
    </figure>
  );
}
