import { useQuery } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { getPairs } from "../lib/api";

type Props = {
  onAnalyze: (base: string, quote: string) => void;
  loading: boolean;
};

export default function PairPicker({ onAnalyze, loading }: Props) {
  const { data: pairs = [], isError } = useQuery({
    queryKey: ["pairs"],
    queryFn: getPairs,
    staleTime: Infinity,
  });

  const [base, setBase] = useState("");
  const [quote, setQuote] = useState("");

  useEffect(() => {
    if (pairs.length > 0 && !base) {
      setBase(pairs[0].base);
      setQuote(pairs[0].quote);
    }
  }, [pairs, base]);

  const bases = [...new Set(pairs.map((p) => p.base))];
  const validQuotes = pairs.filter((p) => p.base === base).map((p) => p.quote);
  const isValid = !!base && !!quote && validQuotes.includes(quote);

  const handleBaseChange = (b: string) => {
    setBase(b);
    const first = pairs.find((p) => p.base === b);
    setQuote(first?.quote ?? "");
  };

  if (isError) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
        Could not load pairs — make sure the backend is running on port 8000.
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-end gap-4 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Base currency
        </label>
        <select
          value={base}
          onChange={(e) => handleBaseChange(e.target.value)}
          className="rounded-lg border border-gray-300 bg-gray-50 px-4 py-2 font-medium text-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {bases.map((b) => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">
          Quote currency
        </label>
        <select
          value={quote}
          onChange={(e) => setQuote(e.target.value)}
          className="rounded-lg border border-gray-300 bg-gray-50 px-4 py-2 font-medium text-gray-800 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {validQuotes.map((q) => (
            <option key={q} value={q}>{q}</option>
          ))}
        </select>
      </div>

      <button
        onClick={() => isValid && onAnalyze(base, quote)}
        disabled={!isValid || loading}
        className="rounded-lg bg-indigo-600 px-6 py-2 font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Analyzing…" : "Analyze"}
      </button>
    </div>
  );
}
