import { useState } from "react";
import { predict } from "./lib/api";
import type { PredictResponse } from "./lib/types";
import PairPicker from "./components/PairPicker";
import VerdictCard from "./components/VerdictCard";
import PredictionPanel from "./components/PredictionPanel";

export default function App() {
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (base: string, quote: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await predict(base, quote);
      setResponse(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-3xl mx-auto px-4 py-10 flex flex-col gap-6">

        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 tracking-tight">ForexOracle</h1>
          <p className="text-gray-500 mt-1">AI-powered forex direction prediction</p>
        </div>

        <PairPicker onAnalyze={handleAnalyze} loading={loading} />

        {loading && (
          <div className="flex justify-center py-10">
            <div className="h-10 w-10 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin" />
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-700">
            {error}
          </div>
        )}

        {response && !loading && (
          <>
            <VerdictCard
              composite={response.composite}
              pair={response.pair}
              currentPrice={response.current_price}
            />
            <PredictionPanel predictions={response.predictions} />

            <div className="rounded-xl border border-dashed border-gray-300 bg-white p-8 text-center text-gray-400 text-sm font-medium">
              Chart (coming soon)
            </div>
            <div className="rounded-xl border border-dashed border-gray-300 bg-white p-8 text-center text-gray-400 text-sm font-medium">
              News (coming soon)
            </div>

            <p className="text-center text-xs text-gray-400">{response.disclaimer}</p>
          </>
        )}
      </div>
    </div>
  );
}
