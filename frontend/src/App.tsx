import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { postPredict } from "./lib/api";
import DisclaimerBanner from "./components/DisclaimerBanner";
import PairPicker from "./components/PairPicker";
import VerdictCard from "./components/VerdictCard";
import AIOverviewCard from "./components/AIOverviewCard";
import PredictionsPanel from "./components/PredictionsPanel";
import PriceChart from "./components/PriceChart";
import NewsSection from "./components/NewsSection";
import MetricsFooter from "./components/MetricsFooter";
import AnalysisProgress from "./components/AnalysisProgress";

type SelectedPair = { base: string; quote: string; pair: string };

export default function App() {
  const [selected, setSelected] = useState<SelectedPair | null>(null);
  const [queryKey, setQueryKey] = useState(0);

  const {
    data: prediction,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["predict", selected?.pair, queryKey],
    queryFn: () => postPredict(selected!.base, selected!.quote),
    enabled: !!selected,
    retry: false,
    staleTime: Infinity,
  });

  const handleAnalyze = (base: string, quote: string) => {
    const next = { base, quote, pair: `${base}${quote}` };
    if (selected?.pair === next.pair) {
      setQueryKey((k) => k + 1);
    } else {
      setSelected(next);
      setQueryKey(0);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <DisclaimerBanner />

      <div className="flex-1 max-w-6xl w-full mx-auto px-4 py-8 flex flex-col gap-5 pb-20">
        <div className="text-center mb-2">
          <h1 className="text-4xl font-extrabold text-gray-900 tracking-tight">ForexOracle</h1>
          <p className="text-gray-500 mt-1 text-sm">AI-powered forex direction prediction</p>
        </div>

        <PairPicker onAnalyze={handleAnalyze} loading={isLoading} />

        {isError && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 flex items-center justify-between gap-4">
            <p className="text-sm text-red-700">
              {error instanceof Error ? error.message : "Prediction failed."}
            </p>
            <button
              onClick={() => refetch()}
              className="shrink-0 rounded-lg border border-red-300 px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-100 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {isLoading && <AnalysisProgress />}

        {prediction && !isLoading && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
              {/* Main column: 2/3 width on large screens */}
              <div className="lg:col-span-2 flex flex-col gap-5">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <VerdictCard
                    composite={prediction.composite}
                    pair={prediction.pair}
                    currentPrice={prediction.current_price}
                  />
                  <PredictionsPanel predictions={prediction.predictions} />
                </div>

                <AIOverviewCard text={prediction.composite.ai_overview} />

                <PriceChart pair={prediction.pair} />
              </div>

              {/* News sidebar: 1/3 width on large screens */}
              <div className="lg:col-span-1">
                <NewsSection articles={prediction.sentiment.articles} />
              </div>
            </div>

            <p className="text-center text-xs text-gray-400 mt-2">{prediction.disclaimer}</p>
          </>
        )}

        {!selected && !isLoading && !prediction && (
          <div className="rounded-xl border border-dashed border-gray-300 bg-white p-12 text-center text-gray-400">
            <p className="text-lg font-medium">Select a currency pair and click Analyze</p>
            <p className="text-sm mt-1">Predictions, chart, and news will appear here.</p>
          </div>
        )}
      </div>

      {selected && <MetricsFooter pair={selected.pair} />}
    </div>
  );
}
