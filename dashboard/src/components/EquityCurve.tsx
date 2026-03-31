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
import type { EquityPoint } from "../types";

interface Props {
  strategyEquity: EquityPoint[];
  initialCapital: number;
  strategyName: string;
  ticker: string;
}

interface ChartPoint {
  date: string;
  strategy: number;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

// Show every Nth tick to avoid crowding
function tickEvery(dates: string[], n: number): Set<string> {
  return new Set(dates.filter((_, i) => i % n === 0));
}

export default function EquityCurve({
  strategyEquity,
  initialCapital,
  strategyName,
  ticker,
}: Props) {
  const data: ChartPoint[] = strategyEquity.map((pt) => ({
    date: pt.date,
    strategy: pt.value,
  }));

  const allDates = data.map((d) => d.date);
  const visibleTicks = tickEvery(allDates, Math.ceil(allDates.length / 8));

  const finalValue = data.at(-1)?.strategy ?? initialCapital;
  const totalReturn = ((finalValue - initialCapital) / initialCapital) * 100;
  const returnColor = totalReturn >= 0 ? "#34d399" : "#f87171";

  return (
    <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-white">Equity Curve</h2>
          <p className="text-xs text-gray-500">
            {strategyName} — {ticker}
          </p>
        </div>
        <span
          className="text-2xl font-bold tabular-nums"
          style={{ color: returnColor }}
        >
          {totalReturn >= 0 ? "+" : ""}
          {totalReturn.toFixed(2)}%
        </span>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickFormatter={(v) => (visibleTicks.has(v) ? v.slice(0, 7) : "")}
            interval={0}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            tickFormatter={(v) => formatCurrency(v)}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={80}
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
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#9ca3af" }}
          />
          <ReferenceLine
            y={initialCapital}
            stroke="#374151"
            strokeDasharray="4 2"
          />
          <Line
            type="monotone"
            dataKey="strategy"
            name={strategyName}
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: "#3b82f6" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
