import { useEffect, useState } from "react";

const STAGES = [
  { label: "Loading market features from database", ms: 0 },
  { label: "Running XGBoost + LSTM inference", ms: 1800 },
  { label: "Fetching financial news (48h)", ms: 4000 },
  { label: "Scoring sentiment with Gemini AI", ms: 6200 },
  { label: "Generating market overview", ms: 8500 },
];

export default function AnalysisProgress() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const timers = STAGES.slice(1).map((stage, i) =>
      setTimeout(() => setActive(i + 1), stage.ms)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div className="rounded-xl border border-indigo-100 bg-white p-6 shadow-sm">
      <div className="flex items-center gap-3 mb-5">
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500" />
        </span>
        <p className="text-sm font-semibold text-gray-700">Analyzing pair…</p>
      </div>
      <ol className="flex flex-col gap-3">
        {STAGES.map((stage, i) => {
          const done = i < active;
          const running = i === active;
          return (
            <li key={i} className="flex items-center gap-3">
              <span className={`shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold transition-colors duration-300
                ${done ? "bg-emerald-500 text-white" : running ? "bg-indigo-500 text-white" : "bg-gray-100 text-gray-400"}`}>
                {done ? "✓" : i + 1}
              </span>
              <span className={`text-sm transition-colors duration-300
                ${done ? "text-gray-400 line-through" : running ? "text-gray-800 font-medium" : "text-gray-400"}`}>
                {stage.label}
              </span>
              {running && (
                <span className="ml-auto flex gap-1">
                  {[0, 1, 2].map((d) => (
                    <span
                      key={d}
                      className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
                      style={{ animationDelay: `${d * 150}ms` }}
                    />
                  ))}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
