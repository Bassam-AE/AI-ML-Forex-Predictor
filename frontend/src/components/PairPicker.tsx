import { useEffect, useState } from "react";
import { getPairs } from "../lib/api";
import type { Pair } from "../lib/types";

type Props = {
  onAnalyze: (base: string, quote: string) => void;
  loading: boolean;
};

export default function PairPicker({ onAnalyze, loading }: Props) {
  const [pairs, setPairs] = useState<Pair[]>([]);
  const [base, setBase] = useState("");
  const [quote, setQuote] = useState("");
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    getPairs()
      .then((data) => {
        setPairs(data);
        if (data[0]) {
          setBase(data[0].base);
          setQuote(data[0].quote);
        }
      })
      .catch((e: unknown) => {
        setFetchError(e instanceof Error ? e.message : "Failed to load pairs");
      });
  }, []);

  const bases = [...new Set(pairs.map((p) => p.base))];
  const validQuotes = pairs.filter((p) => p.base === base).map((p) => p.quote);
  const isValid = base && quote && base !== quote && validQuotes.includes(quote);

  const handleBaseChange = (b: string) => {
    setBase(b);
    const first = pairs.find((p) => p.base === b);
    setQuote(first?.quote ?? "");
  };

  if (fetchError) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
        Could not load pairs from backend: {fetchError}. Make sure the API server is running.
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-end gap-4 p-6 bg-white rounded-xl border border-gray-200 shadow-sm">
      <div className="flex flex-col gap-1">
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Base</label>
        <select
          value={base}
          onChange={(e) => handleBaseChange(e.target.value)}
          className="px-4 py-2 rounded-lg border border-gray-300 bg-gray-50 text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {bases.map((b) => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Quote</label>
        <select
          value={quote}
          onChange={(e) => setQuote(e.target.value)}
          className="px-4 py-2 rounded-lg border border-gray-300 bg-gray-50 text-gray-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {validQuotes.map((q) => (
            <option key={q} value={q}>{q}</option>
          ))}
        </select>
      </div>

      <button
        onClick={() => isValid && onAnalyze(base, quote)}
        disabled={!isValid || loading}
        className="px-6 py-2 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </div>
  );
}
