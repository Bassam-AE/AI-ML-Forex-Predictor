# ForexOracle Frontend

React 18 + TypeScript + Vite single-page app for the ForexOracle prediction dashboard.

## Stack

- React 18 + TypeScript
- Tailwind CSS v3.4
- TanStack Query v5 (data fetching + caching)
- TradingView lightweight-charts v5 (candlestick chart)
- Vite (dev server + build)

## Dev

```bash
npm install
npm run dev        # http://localhost:5173
```

Requires the backend running on port 8000. The Vite dev server proxies `/api/*` → `http://localhost:8000/*` automatically.

## Build

```bash
npm run build      # output → dist/
```

## Components

| Component | Purpose |
|---|---|
| `PairPicker` | Currency pair selector + Analyze button |
| `VerdictCard` | Composite signal, verdict badge, current price |
| `PredictionsPanel` | Probability bars for XGBoost, LSTM, Meta-Learner, Gemini AI |
| `AIOverviewCard` | Gemini-generated market narrative |
| `PriceChart` | 7-day candlestick chart (lightweight-charts v5) |
| `NewsSection` | Scored news articles sidebar with impact badges |
| `MetricsFooter` | Sticky footer with test-set accuracy per model |
| `AnalysisProgress` | Animated stage-by-stage loading indicator |
| `DisclaimerBanner` | Top-of-page disclaimer |
