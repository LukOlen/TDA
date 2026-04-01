import { useState } from "react";
import type { StrategyInfo, CompareRequest } from "../types";

interface Props {
  strategies: StrategyInfo[];
  onSubmit: (req: CompareRequest) => void;
  loading: boolean;
}

interface StrategySlot {
  key: string;
  params: Record<string, number>;
}

const MAX_STRATEGIES = 7;

function defaultParams(info: StrategyInfo): Record<string, number> {
  return Object.fromEntries(info.params.map((p) => [p.name, p.default as number]));
}

export default function CompareConfig({ strategies, onSubmit, loading }: Props) {
  const [ticker, setTicker] = useState("AAPL");
  const [startDate, setStartDate] = useState("2020-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [capital, setCapital] = useState(100_000);
  const [commission, setCommission] = useState(0.001);
  const [slots, setSlots] = useState<StrategySlot[]>(() => {
    const first = strategies[0];
    const second = strategies[1] ?? strategies[0];
    return [
      { key: first.key, params: defaultParams(first) },
      { key: second.key, params: defaultParams(second) },
    ];
  });

  function updateSlotKey(index: number, key: string) {
    const info = strategies.find((s) => s.key === key);
    if (!info) return;
    setSlots((prev) =>
      prev.map((slot, i) => (i === index ? { key, params: defaultParams(info) } : slot))
    );
  }

  function updateSlotParam(index: number, name: string, value: number) {
    setSlots((prev) =>
      prev.map((slot, i) =>
        i === index ? { ...slot, params: { ...slot.params, [name]: value } } : slot
      )
    );
  }

  function addSlot() {
    if (slots.length >= MAX_STRATEGIES) return;
    const info = strategies[slots.length % strategies.length];
    setSlots((prev) => [...prev, { key: info.key, params: defaultParams(info) }]);
  }

  function removeSlot(index: number) {
    if (slots.length <= 2) return;
    setSlots((prev) => prev.filter((_, i) => i !== index));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    onSubmit({
      ticker: ticker.toUpperCase().trim(),
      start_date: startDate,
      end_date: endDate,
      strategies: slots.map((s) => ({ strategy: s.key, params: s.params })),
      initial_capital: capital,
      commission,
    });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-gray-900 rounded-2xl p-6 space-y-5 border border-gray-800"
    >
      <h2 className="text-lg font-semibold text-white">Compare Strategies</h2>

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

      {/* Strategy slots */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-xs text-gray-400">Strategies ({slots.length}/{MAX_STRATEGIES})</label>
          {slots.length < MAX_STRATEGIES && (
            <button
              type="button"
              onClick={addSlot}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              + Add Strategy
            </button>
          )}
        </div>

        {slots.map((slot, index) => {
          const info = strategies.find((s) => s.key === slot.key);
          return (
            <div
              key={index}
              className="bg-gray-800/60 border border-gray-700 rounded-xl p-4 space-y-3"
            >
              <div className="flex items-center gap-3">
                <span className="text-xs text-gray-500 font-mono w-5 shrink-0">#{index + 1}</span>
                <select
                  value={slot.key}
                  onChange={(e) => updateSlotKey(index, e.target.value)}
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {strategies.map((s) => (
                    <option key={s.key} value={s.key}>
                      {s.name}
                    </option>
                  ))}
                </select>
                {slots.length > 2 && (
                  <button
                    type="button"
                    onClick={() => removeSlot(index)}
                    className="text-gray-600 hover:text-red-400 transition-colors text-sm shrink-0"
                    aria-label="Remove strategy"
                  >
                    ✕
                  </button>
                )}
              </div>

              {info && info.params.length > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pl-8">
                  {info.params.map((p) => (
                    <div key={p.name}>
                      <label className="block text-xs text-gray-500 mb-1 capitalize">
                        {p.name.replace(/_/g, " ")}
                      </label>
                      <input
                        type="number"
                        step={p.type === "float" ? "0.1" : "1"}
                        value={slot.params[p.name] ?? p.default}
                        onChange={(e) =>
                          updateSlotParam(
                            index,
                            p.name,
                            p.type === "float"
                              ? parseFloat(e.target.value)
                              : parseInt(e.target.value)
                          )
                        }
                        className="w-full bg-gray-800 border border-gray-600 rounded-lg px-3 py-1.5 text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <button
        type="submit"
        disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-lg py-2.5 text-sm font-semibold text-white"
      >
        {loading ? "Running Comparison…" : "Compare Strategies"}
      </button>
    </form>
  );
}
