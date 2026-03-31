import { useState } from "react";
import type { TradeRecord } from "../types";

interface Props {
  trades: TradeRecord[];
}

type SortKey = keyof TradeRecord;

function formatCurrency(v: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(v);
}

export default function TradeLog({ trades }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("entry_date");
  const [sortAsc, setSortAsc] = useState(true);

  if (trades.length === 0) {
    return (
      <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h2 className="text-lg font-semibold text-white mb-2">Trade Log</h2>
        <p className="text-gray-500 text-sm">No trades were generated.</p>
      </div>
    );
  }

  const sorted = [...trades].sort((a, b) => {
    const va = a[sortKey];
    const vb = b[sortKey];
    if (typeof va === "string" && typeof vb === "string") {
      return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number);
  });

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc((a) => !a);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  function SortIndicator({ col }: { col: SortKey }) {
    if (sortKey !== col) return <span className="opacity-20">↕</span>;
    return <span>{sortAsc ? "↑" : "↓"}</span>;
  }

  const cols: { key: SortKey; label: string }[] = [
    { key: "entry_date", label: "Entry" },
    { key: "exit_date", label: "Exit" },
    { key: "direction", label: "Side" },
    { key: "entry_price", label: "Entry $" },
    { key: "exit_price", label: "Exit $" },
    { key: "pnl", label: "P&L" },
    { key: "pnl_pct", label: "P&L %" },
  ];

  const winners = trades.filter((t) => t.pnl > 0).length;

  return (
    <div className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">
          Trade Log{" "}
          <span className="text-sm text-gray-500 font-normal">
            ({trades.length} trades · {winners} winners)
          </span>
        </h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead>
            <tr className="border-b border-gray-800">
              {cols.map(({ key, label }) => (
                <th
                  key={key}
                  className="px-3 py-2 text-xs text-gray-500 font-medium cursor-pointer hover:text-gray-300 select-none whitespace-nowrap"
                  onClick={() => handleSort(key)}
                >
                  {label} <SortIndicator col={key} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((trade, i) => (
              <tr
                key={i}
                className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors"
              >
                <td className="px-3 py-2 text-gray-300 tabular-nums">
                  {trade.entry_date}
                </td>
                <td className="px-3 py-2 text-gray-300 tabular-nums">
                  {trade.exit_date}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                      trade.direction === "long"
                        ? "bg-blue-900/50 text-blue-300"
                        : "bg-purple-900/50 text-purple-300"
                    }`}
                  >
                    {trade.direction}
                  </span>
                </td>
                <td className="px-3 py-2 text-gray-300 tabular-nums">
                  {formatCurrency(trade.entry_price)}
                </td>
                <td className="px-3 py-2 text-gray-300 tabular-nums">
                  {formatCurrency(trade.exit_price)}
                </td>
                <td
                  className={`px-3 py-2 tabular-nums font-medium ${
                    trade.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {trade.pnl >= 0 ? "+" : ""}
                  {formatCurrency(trade.pnl)}
                </td>
                <td
                  className={`px-3 py-2 tabular-nums font-medium ${
                    trade.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"
                  }`}
                >
                  {trade.pnl_pct >= 0 ? "+" : ""}
                  {trade.pnl_pct.toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
