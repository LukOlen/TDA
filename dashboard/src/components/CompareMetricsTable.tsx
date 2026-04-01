import type { BacktestResult, BacktestMetrics } from "../types";

interface Props {
  results: BacktestResult[];
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

interface MetricDef {
  key: keyof BacktestMetrics;
  label: string;
  format: (v: number) => string;
  higherIsBetter: boolean;
}

const METRICS: MetricDef[] = [
  {
    key: "total_return",
    label: "Total Return",
    format: (v) => `${(v * 100).toFixed(2)}%`,
    higherIsBetter: true,
  },
  {
    key: "cagr",
    label: "CAGR",
    format: (v) => `${(v * 100).toFixed(2)}%`,
    higherIsBetter: true,
  },
  {
    key: "sharpe_ratio",
    label: "Sharpe Ratio",
    format: (v) => v.toFixed(3),
    higherIsBetter: true,
  },
  {
    key: "sortino_ratio",
    label: "Sortino Ratio",
    format: (v) => v.toFixed(3),
    higherIsBetter: true,
  },
  {
    key: "calmar_ratio",
    label: "Calmar Ratio",
    format: (v) => v.toFixed(3),
    higherIsBetter: true,
  },
  {
    key: "max_drawdown",
    label: "Max Drawdown",
    format: (v) => `${(v * 100).toFixed(2)}%`,
    higherIsBetter: false,
  },
  {
    key: "volatility",
    label: "Volatility",
    format: (v) => `${(v * 100).toFixed(2)}%`,
    higherIsBetter: false,
  },
  {
    key: "win_rate",
    label: "Win Rate",
    format: (v) => `${(v * 100).toFixed(1)}%`,
    higherIsBetter: true,
  },
  {
    key: "profit_factor",
    label: "Profit Factor",
    format: (v) => v.toFixed(2),
    higherIsBetter: true,
  },
  {
    key: "avg_trade_return",
    label: "Avg Trade Return",
    format: (v) => `${(v * 100).toFixed(3)}%`,
    higherIsBetter: true,
  },
  {
    key: "total_trades",
    label: "Total Trades",
    format: (v) => v.toFixed(0),
    higherIsBetter: true,
  },
  {
    key: "benchmark_return",
    label: "Benchmark Return",
    format: (v) => `${(v * 100).toFixed(2)}%`,
    higherIsBetter: true,
  },
  {
    key: "alpha",
    label: "Alpha",
    format: (v) => v.toFixed(4),
    higherIsBetter: true,
  },
  {
    key: "beta",
    label: "Beta",
    format: (v) => v.toFixed(3),
    higherIsBetter: false,
  },
];

function bestIndex(values: (number | null)[], higherIsBetter: boolean): number {
  const nums = values.map((v, i) => ({ v, i })).filter((x) => x.v !== null) as {
    v: number;
    i: number;
  }[];
  if (nums.length === 0) return -1;
  return nums.reduce((best, cur) =>
    higherIsBetter ? (cur.v > best.v ? cur : best) : (cur.v < best.v ? cur : best)
  ).i;
}

export default function CompareMetricsTable({ results }: Props) {
  return (
    <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800 overflow-x-auto">
      <h2 className="text-lg font-semibold text-white mb-4">Metrics Comparison</h2>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left text-xs text-gray-500 font-medium pb-3 pr-4 whitespace-nowrap">
              Metric
            </th>
            {results.map((r, i) => (
              <th
                key={r.strategy_name}
                className="text-right text-xs font-medium pb-3 px-3 whitespace-nowrap"
                style={{ color: COLORS[i % COLORS.length] }}
              >
                {r.strategy_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRICS.map((metric) => {
            const values = results.map((r) => r.metrics[metric.key] as number | null);
            const best = bestIndex(values, metric.higherIsBetter);
            return (
              <tr key={metric.key} className="border-t border-gray-800">
                <td className="py-2 pr-4 text-gray-400 whitespace-nowrap">{metric.label}</td>
                {values.map((v, i) => (
                  <td
                    key={i}
                    className={`py-2 px-3 text-right tabular-nums whitespace-nowrap ${
                      i === best ? "text-emerald-400 font-semibold" : "text-gray-300"
                    }`}
                  >
                    {v !== null && v !== undefined ? metric.format(v) : "—"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
