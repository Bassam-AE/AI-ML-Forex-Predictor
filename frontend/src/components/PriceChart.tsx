import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { createChart, CandlestickSeries } from "lightweight-charts";
import { getHistory } from "../lib/api";

type Props = { pair: string };

export default function PriceChart({ pair }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { data: bars, isLoading, isError } = useQuery({
    queryKey: ["history", pair],
    queryFn: () => getHistory(pair, 168),
    staleTime: 5 * 60_000,
  });

  useEffect(() => {
    if (!bars || !containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 350,
      layout: { background: { color: "#ffffff" }, textColor: "#374151" },
      grid: { vertLines: { color: "#f3f4f6" }, horzLines: { color: "#f3f4f6" } },
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderUpColor: "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });

    const formatted = bars.map((b) => ({
      time: (new Date(b.ts).getTime() / 1000) as unknown as import("lightweight-charts").Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    series.setData(formatted);
    chart.timeScale().fitContent();

    const ro = new ResizeObserver(() => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    });
    if (containerRef.current) ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [bars]);

  if (isLoading) {
    return <div className="h-[350px] rounded-xl border border-gray-200 bg-gray-50 animate-pulse" />;
  }

  if (isError) {
    return (
      <div className="h-[350px] rounded-xl border border-red-200 bg-red-50 flex items-center justify-center text-sm text-red-600">
        Failed to load price history.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 px-6 pt-5 pb-3">
        Price History — 7 days
      </p>
      <div ref={containerRef} className="w-full" />
    </div>
  );
}
