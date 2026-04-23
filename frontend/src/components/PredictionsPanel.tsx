import type { PredictResponse } from "../lib/types";

type Props = { predictions: PredictResponse["predictions"] };

const MODELS = [
  { key: "xgboost" as const, label: "XGBoost", ai: false },
  { key: "lstm" as const, label: "LSTM", ai: false },
  { key: "meta_learner" as const, label: "Meta-Learner", ai: false },
  { key: "gemini" as const, label: "Gemini AI", ai: true },
];

function barColor(p: number): string {
  if (p >= 0.65) return "bg-emerald-600";
  if (p >= 0.55) return "bg-emerald-400";
  if (p <= 0.35) return "bg-rose-600";
  if (p <= 0.45) return "bg-rose-400";
  return "bg-gray-400";
}

export default function PredictionsPanel({ predictions }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-5">
        Model Predictions
      </p>
      <div className="flex flex-col gap-5">
        {MODELS.map(({ key, label, ai }) => {
          const prob = predictions[key]?.prob_up ?? 0.5;
          return (
            <div key={key}>
              {ai && <hr className="border-gray-100 -mt-1 mb-4" />}
              <div className="flex justify-between text-sm font-medium text-gray-700 mb-1.5">
                <span className="flex items-center gap-1.5">
                  {label}
                  {ai && (
                    <span className="rounded px-1.5 py-0.5 text-[10px] font-semibold bg-violet-100 text-violet-600 leading-none">
                      LLM
                    </span>
                  )}
                </span>
                <span className="tabular-nums">{prob.toFixed(3)}</span>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${barColor(prob)}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                <span>0%</span>
                <span className="text-gray-300">|50%</span>
                <span>100%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
