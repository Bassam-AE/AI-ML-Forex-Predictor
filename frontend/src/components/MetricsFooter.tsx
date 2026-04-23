import { useQuery } from "@tanstack/react-query";
import { getMetrics } from "../lib/api";

type Props = { pair: string };

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

export default function MetricsFooter({ pair }: Props) {
  const { data } = useQuery({
    queryKey: ["metrics", pair],
    queryFn: () => getMetrics(pair),
    staleTime: Infinity,
  });

  if (!data) return null;

  return (
    <div className="sticky bottom-0 z-40 bg-slate-900 text-slate-300 text-xs px-4 py-2.5 flex flex-wrap items-center justify-center gap-x-4 gap-y-1">
      <span className="font-semibold text-slate-100">Test accuracy for {pair}:</span>
      <span>XGBoost <span className="text-slate-100 font-medium">{pct(data.xgboost.accuracy)}</span></span>
      <span className="text-slate-600">·</span>
      <span>LSTM <span className="text-slate-100 font-medium">{pct(data.lstm.accuracy)}</span></span>
      <span className="text-slate-600">·</span>
      <span>Meta <span className="text-slate-100 font-medium">{pct(data.meta_learner.accuracy)}</span></span>
      <span className="text-slate-600">·</span>
      <span>Always-Up baseline <span className="text-slate-100 font-medium">{pct(data.baselines.always_up.accuracy)}</span></span>
      <span className="text-slate-600 hidden sm:inline">·</span>
      <span className="hidden sm:inline">n={data.test_samples.toLocaleString()}</span>
    </div>
  );
}
