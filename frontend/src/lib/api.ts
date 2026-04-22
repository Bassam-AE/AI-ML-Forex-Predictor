import type { OHLCBar, Pair, PredictResponse } from "./types";

// In dev, Vite proxies /api → http://localhost:8000 (no CORS needed).
// In production set VITE_API_URL to the deployed backend URL.
export const API_URL = import.meta.env.VITE_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
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

export const predict = (base: string, quote: string): Promise<PredictResponse> =>
  request<PredictResponse>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base, quote }),
  });
