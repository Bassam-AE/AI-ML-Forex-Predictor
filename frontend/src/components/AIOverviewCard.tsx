type Props = { text: string };

export default function AIOverviewCard({ text }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-3">
        AI Overview
      </p>
      <p className="text-gray-700 leading-relaxed text-sm">
        {text || "No overview available."}
      </p>
    </div>
  );
}
