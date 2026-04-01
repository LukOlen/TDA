import { useState } from "react";
import { useStrategies, useBacktest, useCompare } from "./hooks/useBacktest";
import StrategyConfig from "./components/StrategyConfig";
import EquityCurve from "./components/EquityCurve";
import MetricsPanel from "./components/MetricsPanel";
import TradeLog from "./components/TradeLog";
import CompareConfig from "./components/CompareConfig";
import CompareEquityCurve from "./components/CompareEquityCurve";
import CompareMetricsTable from "./components/CompareMetricsTable";

type Mode = "single" | "compare";

export default function App() {
  const [mode, setMode] = useState<Mode>("single");
  const { strategies, loading: strategiesLoading, error: strategiesError } = useStrategies();
  const { result, loading: backtestLoading, error: backtestError, run } = useBacktest();
  const {
    result: compareResult,
    loading: compareLoading,
    error: compareError,
    run: runCompare,
  } = useCompare();

  if (strategiesLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-gray-400">Connecting to API…</p>
      </div>
    );
  }

  if (strategiesError) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-2">
          <p className="text-red-400 font-medium">Failed to connect to API</p>
          <p className="text-gray-500 text-sm">{strategiesError}</p>
          <p className="text-gray-600 text-xs">
            Make sure the API is running: <code>uvicorn api.main:app --reload</code>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-white tracking-tight">
              TDA Backtester
            </h1>
            <p className="text-xs text-gray-500">
              Algorithmic Trading &amp; Statistical Analysis
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Mode tabs */}
            <div className="flex items-center bg-gray-900 rounded-lg p-1 border border-gray-800">
              <button
                onClick={() => setMode("single")}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  mode === "single"
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Single
              </button>
              <button
                onClick={() => setMode("compare")}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  mode === "compare"
                    ? "bg-blue-600 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                Compare
              </button>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-gray-400">API Online</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {mode === "single" ? (
          <>
            <StrategyConfig
              strategies={strategies}
              onSubmit={run}
              loading={backtestLoading}
            />

            {backtestError && (
              <div className="bg-red-950/50 border border-red-800 rounded-xl px-4 py-3 text-red-300 text-sm">
                {backtestError}
              </div>
            )}

            {backtestLoading && (
              <div className="space-y-4 animate-pulse">
                <div className="h-80 bg-gray-900 rounded-2xl border border-gray-800" />
                <div className="h-32 bg-gray-900 rounded-2xl border border-gray-800" />
              </div>
            )}

            {result && !backtestLoading && (
              <>
                <EquityCurve
                  strategyEquity={result.equity_curve}
                  initialCapital={result.initial_capital}
                  strategyName={result.strategy_name}
                  ticker={result.ticker}
                />
                <MetricsPanel metrics={result.metrics} />
                <TradeLog trades={result.trades} />
              </>
            )}

            {!result && !backtestLoading && (
              <div className="flex flex-col items-center justify-center py-24 space-y-3 text-center">
                <div className="text-5xl">📈</div>
                <p className="text-gray-400 font-medium">
                  Configure a strategy above and run your first backtest.
                </p>
                <p className="text-gray-600 text-sm max-w-md">
                  The engine fetches live OHLCV data from Yahoo Finance, runs a
                  vectorised simulation, and returns equity curves, performance
                  statistics, and a full trade log.
                </p>
              </div>
            )}
          </>
        ) : (
          <>
            <CompareConfig
              strategies={strategies}
              onSubmit={runCompare}
              loading={compareLoading}
            />

            {compareError && (
              <div className="bg-red-950/50 border border-red-800 rounded-xl px-4 py-3 text-red-300 text-sm">
                {compareError}
              </div>
            )}

            {compareLoading && (
              <div className="space-y-4 animate-pulse">
                <div className="h-96 bg-gray-900 rounded-2xl border border-gray-800" />
                <div className="h-64 bg-gray-900 rounded-2xl border border-gray-800" />
              </div>
            )}

            {compareResult && !compareLoading && (
              <>
                <CompareEquityCurve
                  results={compareResult.results}
                  initialCapital={compareResult.results[0]?.initial_capital ?? 100_000}
                  ticker={compareResult.ticker}
                />
                <CompareMetricsTable results={compareResult.results} />
              </>
            )}

            {!compareResult && !compareLoading && (
              <div className="flex flex-col items-center justify-center py-24 space-y-3 text-center">
                <div className="text-5xl">⚖️</div>
                <p className="text-gray-400 font-medium">
                  Select 2 or more strategies above and compare their performance.
                </p>
                <p className="text-gray-600 text-sm max-w-md">
                  All strategies are run on the same ticker and date range so you
                  can see their equity curves and metrics side by side.
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
