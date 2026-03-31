import { useState, useEffect } from "react";
import type { StrategyInfo, BacktestRequest } from "../types";

interface Props {
  strategies: StrategyInfo[];
  onSubmit: (req: BacktestRequest) => void;
  loading: boolean;
}

const DEFAULT_PARAMS: Record<string, Record<string, number>> = {
  sma_crossover: { fast_period: 20, slow_period: 50 },
  mean_reversion: { period: 20, num_std: 2.0 },
  momentum: { lookback: 20 },
};

export default function StrategyConfig({ strategies, onSubmit, loading }: Props) {
  const [ticker, setTicker] = useState("AAPL");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [strategyKey, setStrategyKey] = useState(strategies[0]?.key ?? "sma_crossover");
  const [capital, setCapital] = useState(100_000);
  const [commission, setCommission] = useState(0.001);
  const [params, setParams] = useState<Record<string, number>>(
    DEFAULT_PARAMS[strategyKey] ?? {}
  );

  const selectedStrategy = strategies.find((s) => s.key === strategyKey);

  useEffect(() => {
    setParams(DEFAULT_PARAMS[strategyKey] ?? {});
  }, [strategyKey]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      ticker: ticker.toUpperCase().trim(),
      start_date: startDate,
      end_date: endDate,
      strategy: strategyKey,
      initial_capital: capital,
      commission,
      params,
    });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-gray-900 rounded-2xl p-6 space-y-5 border border-gray-800"
    >
      <h2 className="text-lg font-semibold text-white">Backtest Configuration</h2>

      {/* Ticker + Dates */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Ticker</label>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="AAPL"
            required
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>
      </div>

      {/* Strategy */}
      <div>
        <label className="block text-xs text-gray-400 mb-1">Strategy</label>
        <select
          value={strategyKey}
          onChange={(e) => setStrategyKey(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {strategies.map((s) => (
            <option key={s.key} value={s.key}>
              {s.name}
            </option>
          ))}
        </select>
        {selectedStrategy && (
          <p className="mt-1 text-xs text-gray-500">{selectedStrategy.description}</p>
        )}
      </div>

      {/* Dynamic strategy params */}
      {selectedStrategy && selectedStrategy.params.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {selectedStrategy.params.map((p) => (
            <div key={p.name}>
              <label className="block text-xs text-gray-400 mb-1 capitalize">
                {p.name.replace(/_/g, " ")}
              </label>
              <input
                type="number"
                step={p.type === "float" ? "0.1" : "1"}
                value={params[p.name] ?? p.default}
                onChange={(e) =>
                  setParams((prev) => ({
                    ...prev,
                    [p.name]: p.type === "float" ? parseFloat(e.target.value) : parseInt(e.target.value),
                  }))
                }
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="mt-0.5 text-xs text-gray-600">{p.description}</p>
            </div>
          ))}
        </div>
      )}

      {/* Capital + Commission */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-gray-400 mb-1">Initial Capital ($)</label>
          <input
            type="number"
            value={capital}
            onChange={(e) => setCapital(parseFloat(e.target.value))}
            min={1000}
            step={1000}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">Commission (%)</label>
          <input
            type="number"
            value={(commission * 100).toFixed(2)}
            onChange={(e) => setCommission(parseFloat(e.target.value) / 100)}
            min={0}
            max={5}
            step={0.01}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-lg py-2.5 text-sm font-semibold text-white"
      >
        {loading ? "Running Backtest…" : "Run Backtest"}
      </button>
    </form>
  );
}
