import type { BacktestRequest, BacktestResult, CompareRequest, CompareResponse, StrategyInfo } from "../types";

const BASE = "/api";

export async function fetchStrategies(): Promise<StrategyInfo[]> {
  const res = await fetch(`${BASE}/strategies`);
  if (!res.ok) throw new Error(`Failed to fetch strategies: ${res.statusText}`);
  return res.json();
}

export async function runBacktest(
  req: BacktestRequest
): Promise<BacktestResult> {
  const res = await fetch(`${BASE}/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Backtest failed");
  }
  return res.json();
}

export async function compareStrategies(
  req: CompareRequest
): Promise<CompareResponse> {
  const res = await fetch(`${BASE}/backtest/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Comparison failed");
  }
  return res.json();
}
