import type { NewsArticle } from "../lib/types";

type Props = { articles: NewsArticle[] };

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const h = Math.floor(diff / 3_600_000);
  if (h < 1) return `${Math.floor(diff / 60_000)}m ago`;
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function impactBadge(score: number) {
  if (score > 0.15) return { cls: "bg-emerald-100 text-emerald-700", label: `+${score.toFixed(2)}` };
  if (score < -0.15) return { cls: "bg-rose-100 text-rose-700", label: score.toFixed(2) };
  return { cls: "bg-gray-100 text-gray-500", label: score.toFixed(2) };
}

export default function NewsSection({ articles }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-4">
        Recent News
      </p>
      {articles.length === 0 ? (
        <p className="text-sm text-gray-400">No recent news available.</p>
      ) : (
        <div className="flex flex-col divide-y divide-gray-100">
          {articles.map((a, i) => {
            const badge = impactBadge(a.impact_score);
            return (
              <div key={i} className="py-4 first:pt-0 last:pb-0">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <a
                      href={a.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium text-gray-800 hover:text-indigo-600 leading-snug line-clamp-2"
                    >
                      {a.title}
                    </a>
                    <p className="text-xs text-gray-400 mt-1">
                      {a.source} · {relativeTime(a.published_at)}
                    </p>
                    {a.reasoning && (
                      <p className="text-xs text-gray-500 mt-1.5 italic">{a.reasoning}</p>
                    )}
                  </div>
                  <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold tabular-nums ${badge.cls}`}>
                    {badge.label}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
