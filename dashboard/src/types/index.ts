export interface EquityPoint {
  date: string;
  value: number;
}

export interface TradeRecord {
  entry_date: string;
  exit_date: string;
  direction: "long" | "short";
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
}

export interface BacktestMetrics {
  total_return: number | null;
  cagr: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  max_drawdown: number | null;
  calmar_ratio: number | null;
  volatility: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  avg_trade_return: number | null;
  total_trades: number;
  benchmark_return: number | null;
  beta: number | null;
  alpha: number | null;
}

export interface BacktestResult {
  strategy_name: string;
  ticker: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: TradeRecord[];
}

export interface StrategyParam {
  name: string;
  type: "int" | "float";
  default: number;
  description: string;
}

export interface StrategyInfo {
  key: string;
  name: string;
  description: string;
  params: StrategyParam[];
}

export interface BacktestRequest {
  ticker: string;
  start_date: string;
  end_date: string;
  strategy: string;
  initial_capital: number;
  commission: number;
  params: Record<string, number>;
}

export interface StrategySpec {
  strategy: string;
  params: Record<string, number>;
}

export interface CompareRequest {
  ticker: string;
  start_date: string;
  end_date: string;
  strategies: StrategySpec[];
  initial_capital: number;
  commission: number;
}

export interface CompareResponse {
  ticker: string;
  start_date: string;
  end_date: string;
  results: BacktestResult[];
}
