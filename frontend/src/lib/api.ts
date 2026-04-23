import type { MetricsResponse, OHLCBar, Pair, PredictResponse } from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const getPairs = (): Promise<Pair[]> =>
  request<Pair[]>("/pairs");

export const getHistory = (pair: string, hours = 168): Promise<OHLCBar[]> =>
  request<OHLCBar[]>(`/history/${pair}?hours=${hours}`);

export const postPredict = (base: string, quote: string): Promise<PredictResponse> =>
  request<PredictResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base, quote }),
  });

export const getMetrics = (pair: string): Promise<MetricsResponse> =>
  request<MetricsResponse>(`/metrics/${pair}`);
