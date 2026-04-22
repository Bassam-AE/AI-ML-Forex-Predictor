import type { PredictResponse } from "../lib/types";

type Props = {
  predictions: PredictResponse["predictions"];
};

const MODELS = [
  { key: "xgboost", label: "XGBoost" },
  { key: "lstm", label: "LSTM" },
  { key: "meta_learner", label: "Meta Learner" },
] as const;

function barColor(p: number) {
  if (p > 0.55) return "bg-green-500";
  if (p < 0.45) return "bg-red-500";
  return "bg-gray-400";
}

export default function PredictionPanel({ predictions }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Model Predictions
      </h2>
      <div className="flex flex-col gap-4">
        {MODELS.map(({ key, label }) => {
          const prob = predictions[key].prob_up;
          return (
            <div key={key}>
              <div className="flex justify-between text-sm font-medium text-gray-700 mb-1">
                <span>{label}</span>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
              <div className="h-3 w-full rounded-full bg-gray-100 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${barColor(prob)}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
