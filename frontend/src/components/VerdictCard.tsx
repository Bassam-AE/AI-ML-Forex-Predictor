import type { PredictResponse } from "../lib/types";

type Props = {
  composite: PredictResponse["composite"];
  pair: string;
  currentPrice: number;
};

const STYLES = {
  bullish: "bg-emerald-500",
  bearish: "bg-rose-500",
  neutral: "bg-slate-500",
};

const LABELS = {
  bullish: "↑ BULLISH",
  bearish: "↓ BEARISH",
  neutral: "→ NEUTRAL",
};

export default function VerdictCard({ composite, pair, currentPrice }: Props) {
  return (
    <div className={`rounded-xl p-6 text-white shadow-sm ${STYLES[composite.verdict]}`}>
      <div className="flex items-start justify-between mb-2">
        <div>
          <p className="text-sm font-semibold uppercase tracking-widest opacity-80">{pair}</p>
          <p className="text-2xl font-bold mt-0.5">{currentPrice.toFixed(4)}</p>
        </div>
        <span className="rounded-full bg-white/20 px-4 py-1.5 text-lg font-bold">
          {LABELS[composite.verdict]}
        </span>
      </div>
      <p className="text-5xl font-extrabold tracking-tight mt-3">
        {(composite.prob_up * 100).toFixed(1)}%
      </p>
      <p className="text-sm font-medium opacity-75 mt-1">composite probability of upward move</p>
    </div>
  );
}
