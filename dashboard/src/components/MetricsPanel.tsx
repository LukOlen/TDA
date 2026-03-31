import type { BacktestMetrics } from "../types";

interface Props {
  metrics: BacktestMetrics;
}

interface MetricCard {
  label: string;
  value: string;
  positive?: boolean | null;
  description: string;
}

function fmtPct(v: number | null, decimals = 2): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(decimals)}%`;
}

function fmtNum(v: number | null, decimals = 3): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(decimals);
}

function isPositive(v: number | null, invert = false): boolean | null {
  if (v === null) return null;
  const result = v > 0;
  return invert ? !result : result;
}

export default function MetricsPanel({ metrics }: Props) {
  const cards: MetricCard[] = [
    {
      label: "Total Return",
      value: fmtPct(metrics.total_return),
      positive: isPositive(metrics.total_return),
      description: "Cumulative return over the period",
    },
    {
      label: "CAGR",
      value: fmtPct(metrics.cagr),
      positive: isPositive(metrics.cagr),
      description: "Compound annual growth rate",
    },
    {
      label: "Sharpe Ratio",
      value: fmtNum(metrics.sharpe_ratio),
      positive: metrics.sharpe_ratio !== null ? metrics.sharpe_ratio > 1 : null,
      description: "Risk-adjusted return (>1 is good)",
    },
    {
      label: "Sortino Ratio",
      value: fmtNum(metrics.sortino_ratio),
      positive: metrics.sortino_ratio !== null ? metrics.sortino_ratio > 1 : null,
      description: "Downside risk-adjusted return",
    },
    {
      label: "Max Drawdown",
      value: fmtPct(metrics.max_drawdown),
      positive: isPositive(metrics.max_drawdown, true),
      description: "Largest peak-to-trough decline",
    },
    {
      label: "Calmar Ratio",
      value: fmtNum(metrics.calmar_ratio),
      positive: metrics.calmar_ratio !== null ? metrics.calmar_ratio > 0.5 : null,
      description: "CAGR / |Max Drawdown|",
    },
    {
      label: "Volatility",
      value: fmtPct(metrics.volatility),
      positive: null,
      description: "Annualised standard deviation",
    },
    {
      label: "Win Rate",
      value: fmtPct(metrics.win_rate),
      positive: metrics.win_rate !== null ? metrics.win_rate > 0.5 : null,
      description: "Fraction of winning trades",
    },
    {
      label: "Profit Factor",
      value: fmtNum(metrics.profit_factor),
      positive: metrics.profit_factor !== null ? metrics.profit_factor > 1 : null,
      description: "Gross profit / gross loss",
    },
    {
      label: "Avg Trade Return",
      value: fmtPct(metrics.avg_trade_return),
      positive: isPositive(metrics.avg_trade_return),
      description: "Mean P&L per trade",
    },
    {
      label: "Total Trades",
      value: String(metrics.total_trades),
      positive: null,
      description: "Number of completed trades",
    },
    {
      label: "Benchmark Return",
      value: fmtPct(metrics.benchmark_return),
      positive: null,
      description: "Buy-and-hold return",
    },
    {
      label: "Beta",
      value: fmtNum(metrics.beta),
      positive: null,
      description: "Market beta vs buy-and-hold",
    },
    {
      label: "Alpha",
      value: fmtPct(metrics.alpha),
      positive: isPositive(metrics.alpha),
      description: "Annualised Jensen's alpha",
    },
  ];

  return (
    <div>
      <h2 className="text-lg font-semibold text-white mb-4">Performance Metrics</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-3">
        {cards.map(({ label, value, positive, description }) => (
          <div
            key={label}
            title={description}
            className="bg-gray-900 border border-gray-800 rounded-xl p-3 flex flex-col gap-1"
          >
            <span className="text-xs text-gray-500 truncate">{label}</span>
            <span
              className={`text-base font-bold tabular-nums ${
                positive === null
                  ? "text-white"
                  : positive
                  ? "text-emerald-400"
                  : "text-red-400"
              }`}
            >
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
