import { useStrategies, useBacktest } from "./hooks/useBacktest";
import StrategyConfig from "./components/StrategyConfig";
import EquityCurve from "./components/EquityCurve";
import MetricsPanel from "./components/MetricsPanel";
import TradeLog from "./components/TradeLog";

export default function App() {
  const { strategies, loading: strategiesLoading, error: strategiesError } = useStrategies();
  const { result, loading: backtestLoading, error: backtestError, run } = useBacktest();

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
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs text-gray-400">API Online</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Config form */}
        <StrategyConfig
          strategies={strategies}
          onSubmit={run}
          loading={backtestLoading}
        />

        {/* Backtest error */}
        {backtestError && (
          <div className="bg-red-950/50 border border-red-800 rounded-xl px-4 py-3 text-red-300 text-sm">
            {backtestError}
          </div>
        )}

        {/* Loading skeleton */}
        {backtestLoading && (
          <div className="space-y-4 animate-pulse">
            <div className="h-80 bg-gray-900 rounded-2xl border border-gray-800" />
            <div className="h-32 bg-gray-900 rounded-2xl border border-gray-800" />
          </div>
        )}

        {/* Results */}
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

        {/* Empty state */}
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
      </main>
    </div>
  );
}
