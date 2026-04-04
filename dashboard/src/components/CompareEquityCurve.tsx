import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import type { BacktestResult } from "../types";

interface Props {
  results: BacktestResult[];
  initialCapital: number;
  ticker: string;
}

const COLORS = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#06b6d4",
  "#f97316",
];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function tickEvery(dates: string[], n: number): Set<string> {
  return new Set(dates.filter((_, i) => i % n === 0));
}

export default function CompareEquityCurve({ results, initialCapital, ticker }: Props) {
  // Build merged chart data aligned by date index using the longest series
  const longest = results.reduce(
    (a, b) => (a.equity_curve.length >= b.equity_curve.length ? a : b),
    results[0]
  );

  const data = longest.equity_curve.map((pt, i) => {
    const row: Record<string, string | number> = { date: pt.date };
    results.forEach((r) => {
      const val = r.equity_curve[i]?.value;
      if (val !== undefined) row[r.strategy_name] = val;
    });
    return row;
  });

  const allDates = data.map((d) => d.date as string);
  const visibleTicks = tickEvery(allDates, Math.ceil(allDates.length / 8));

  return (
    <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-white">Equity Curves</h2>
        <p className="text-xs text-gray-500">{ticker} — strategy comparison</p>
      </div>

      {/* Return badges */}
      <div className="flex flex-wrap gap-3 mb-5">
        {results.map((r, i) => {
          const final = r.equity_curve.at(-1)?.value ?? initialCapital;
          const ret = ((final - initialCapital) / initialCapital) * 100;
          return (
            <div key={r.strategy_name} className="flex items-center gap-2">
              <span
                className="inline-block w-3 h-3 rounded-full shrink-0"
                style={{ backgroundColor: COLORS[i % COLORS.length] }}
              />
              <span className="text-xs text-gray-400">{r.strategy_name}</span>
              <span
                className="text-xs font-bold tabular-nums"
                style={{ color: ret >= 0 ? "#34d399" : "#f87171" }}
              >
                {ret >= 0 ? "+" : ""}
                {ret.toFixed(2)}%
              </span>
            </div>
          );
        })}
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickFormatter={(v) => (visibleTicks.has(v) ? (v as string).slice(0, 7) : "")}
            interval={0}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v) => formatCurrency(v)}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={85}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              border: "1px solid #374151",
              borderRadius: 8,
            }}
            labelStyle={{ color: "#9ca3af", fontSize: 12 }}
            formatter={(value: number) => [formatCurrency(value)]}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "#9ca3af" }} />
          <ReferenceLine y={initialCapital} stroke="#374151" strokeDasharray="4 2" />
          {results.map((r, i) => (
            <Line
              key={r.strategy_name}
              type="monotone"
              dataKey={r.strategy_name}
              stroke={COLORS[i % COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
