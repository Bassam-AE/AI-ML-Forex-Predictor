import type { PredictResponse } from "../lib/types";

type Props = {
  composite: PredictResponse["composite"];
  pair: string;
  currentPrice: number;
};

const STYLES = {
  bullish: {
    bg: "bg-green-50 border-green-200",
    badge: "bg-green-100 text-green-800",
    label: "↑ BULLISH",
  },
  bearish: {
    bg: "bg-red-50 border-red-200",
    badge: "bg-red-100 text-red-800",
    label: "↓ BEARISH",
  },
  neutral: {
    bg: "bg-gray-50 border-gray-200",
    badge: "bg-gray-100 text-gray-700",
    label: "→ NEUTRAL",
  },
};

export default function VerdictCard({ composite, pair, currentPrice }: Props) {
  const s = STYLES[composite.verdict];
  return (
    <div className={`rounded-xl border p-6 ${s.bg}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-sm font-semibold text-gray-500 uppercase tracking-wide">{pair}</span>
          <p className="text-2xl font-bold text-gray-800 mt-0.5">{currentPrice.toFixed(4)}</p>
        </div>
        <span className={`text-lg font-bold px-4 py-1.5 rounded-full ${s.badge}`}>
          {s.label}
        </span>
      </div>
      <p className="text-3xl font-bold text-gray-800 mb-3">
        {(composite.prob_up * 100).toFixed(1)}% chance up
      </p>
      <p className="text-sm text-gray-600 leading-relaxed">{composite.ai_overview}</p>
    </div>
  );
}
