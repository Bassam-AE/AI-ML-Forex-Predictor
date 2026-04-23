# ForexOracle — Technical Reference

## Stack

| Layer                  | Technology                                                               |
| ---------------------- | ------------------------------------------------------------------------ |
| Language               | Python 3.14                                                              |
| Package manager        | uv                                                                       |
| Database               | SQLite (per-pair OHLC tables)                                            |
| ML — tabular           | XGBoost 3.x                                                              |
| ML — sequence          | PyTorch 2.x LSTM                                                         |
| ML — ensemble          | scikit-learn LogisticRegression                                          |
| Backend                | FastAPI + uvicorn                                                        |
| Validation             | Pydantic v2 + pydantic-settings                                          |
| News                   | NewsAPI v2 (pair-specific AND queries, 30-min in-memory cache)           |
| AI signals             | Google Gemini 2.5 Flash (batch sentiment · directional probe · overview) |
| Frontend               | React 18 + TypeScript + Vite + Tailwind CSS v3.4                         |
| Chart                  | TradingView lightweight-charts v5                                        |
| HTTP client (frontend) | TanStack Query v5                                                        |

---

## Repository Layout

```
.
├── app/
│   ├── ai/
│   │   ├── news.py         # Async NewsAPI client; pair-specific AND queries; 30-min cache
│   │   ├── sentiment.py    # Gemini batch scorer (1 API call per pair, not per article)
│   │   ├── ai_predict.py   # Gemini directional probe → prob_up float
│   │   ├── overview.py     # Gemini 2-3 sentence market summary
│   │   ├── aggregate.py    # Confidence-weighted sentiment aggregation
│   │   └── schemas.py      # RawArticle, ScoredArticle, GeminiBatchOutput
│   ├── api/
│   │   ├── main.py         # FastAPI app + CORS + model lifespan loader
│   │   ├── schemas.py      # Pydantic request/response models
│   │   └── routes/
│   │       ├── health.py
│   │       ├── pairs.py
│   │       ├── history.py  # OHLC history for chart
│   │       ├── predict.py  # Main prediction endpoint
│   │       └── metrics.py  # Per-pair test accuracy metrics
│   ├── data/
│   │   ├── fetch.py        # yfinance → SQLite (2y hourly OHLC)
│   │   └── split_data.py   # Chronological split → {pair}_TRAIN / {pair}_TEST
│   ├── evaluate/
│   │   └── evaluate_holdout.py
│   ├── features/
│   │   └── tabular.py      # 18 manual technical indicators (no pandas-ta)
│   ├── models/
│   │   └── lstm_model.py   # ForexLSTM nn.Module
│   ├── serving/
│   │   └── model_loader.py # Loads all models at startup; predict_for_pair()
│   ├── train/
│   │   ├── train_xgb.py
│   │   ├── train_lstm.py   # Also saves val/test .npy prediction files
│   │   └── train_meta.py   # No torch import (avoids macOS OpenMP deadlock)
│   ├── config.py           # pydantic-settings: DB_PATH, API keys, SUPPORTED_PAIRS
│   └── db.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── PairPicker.tsx
│   │   │   ├── VerdictCard.tsx         # Composite signal + price
│   │   │   ├── PredictionsPanel.tsx    # Probability bars (XGB, LSTM, Meta, Gemini)
│   │   │   ├── AIOverviewCard.tsx      # Gemini narrative
│   │   │   ├── PriceChart.tsx          # lightweight-charts v5 candlestick
│   │   │   ├── NewsSection.tsx         # Scored articles sidebar
│   │   │   ├── MetricsFooter.tsx       # Sticky test-accuracy footer
│   │   │   ├── AnalysisProgress.tsx    # Animated loading checklist
│   │   │   └── DisclaimerBanner.tsx
│   │   ├── lib/
│   │   │   ├── api.ts      # Typed API client (Vite /api proxy in dev)
│   │   │   └── types.ts    # PredictResponse, MetricsResponse, etc.
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── vite.config.ts      # /api → http://localhost:8000 proxy
├── models/
│   └── {EURUSD,GBPUSD,USDINR}/
│       ├── xgb.json
│       ├── lstm.pt
│       ├── lstm_scaler.pkl         # {"mean": np.array(5), "std": np.array(5)}
│       ├── lstm_val_probs.npy      # Pre-computed val predictions (for train_meta)
│       ├── lstm_val_labels.npy
│       ├── lstm_test_probs.npy
│       ├── lstm_test_labels.npy
│       ├── meta.pkl
│       ├── metrics.json
│       └── holdout_predictions.csv
├── scripts/
│   ├── smoke_test.py               # Load all 3 models for EURUSD, print prob table
│   └── test_news_pipeline.py       # End-to-end news + Gemini sentiment test
├── .env                            # GEMINI_API_KEY, NEWS_API_KEY (not committed)
├── .env.example
├── pyproject.toml
└── uv.lock
```

---

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) installed
- Node.js 18+
- A NewsAPI key (free tier: 100 req/day) — newsapi.org
- A Google Gemini API key — aistudio.google.com

---

## First-Time Setup

```bash
# Install Python dependencies (creates .venv automatically)
uv sync

# Install the app package as editable (required for app.* imports)
uv pip install -e .

# Install frontend dependencies
cd frontend && npm install && cd ..

# Copy and fill in API keys
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and NEWS_API_KEY
```

---

## Full Training Pipeline

Run in order — each step depends on the previous. Trained weights are already committed in `models/` so this step is optional.

```bash
# 1. Fetch 2 years of hourly OHLC data into SQLite
uv run python -m app.data.fetch

# 2. Split into train (all but last 10 days) and test (last 10 days)
uv run python -m app.data.split_data

# 3. Train XGBoost — saves xgb.json + metrics.json per pair
uv run python -m app.train.train_xgb

# 4. Train LSTM — saves lstm.pt, scaler, and val/test .npy files per pair
#    ~3-5 min on CPU
uv run python -m app.train.train_lstm

# 5. Train meta-learner — stacks XGBoost + LSTM val predictions via LogisticRegression
uv run python -m app.train.train_meta

# 6. (Optional) Evaluate on the held-out 10-day test split
uv run python -m app.evaluate.evaluate_holdout
```

---

## Running the App

Terminal 1 — backend:

```bash
uv run uvicorn app.api.main:app --reload
# http://localhost:8000
```

Terminal 2 — frontend:

```bash
cd frontend && npm run dev
# http://localhost:5173
```

---

## Verification

```bash
# Confirm all 3 models load and produce predictions for EURUSD
uv run python scripts/smoke_test.py

# Confirm news fetch + Gemini sentiment pipeline works end-to-end
uv run python scripts/test_news_pipeline.py
```

---

## API Reference

Base URL in dev: `http://localhost:8000`  
Frontend proxies `/api/*` → `http://localhost:8000/*` via Vite.

### GET /health

```json
{ "status": "ok" }
```

### GET /pairs

```json
[{"pair": "EURUSD", "base": "EUR", "quote": "USD"}, ...]
```

### GET /history/{pair}?hours=168

Returns up to `hours` of hourly OHLC bars (max 1000). `pair` ∈ `{EURUSD, GBPUSD, USDINR}`.

### GET /metrics/{pair}

```json
{
  "pair": "EURUSD",
  "test_samples": 240,
  "xgboost": { "accuracy": 0.529, "logloss": 0.693 },
  "lstm": { "accuracy": 0.521, "logloss": 0.698 },
  "meta_learner": { "accuracy": 0.533, "logloss": 0.691 },
  "baselines": {
    "always_up": { "accuracy": 0.504 },
    "previous_direction": { "accuracy": 0.496 }
  }
}
```

### POST /predict

```json
// Request
{"base": "EUR", "quote": "USD"}

// Response
{
  "pair": "EURUSD",
  "timestamp": "2026-04-23T01:47:17Z",
  "current_price": 1.1677,
  "predictions": {
    "xgboost":      {"prob_up": 0.4701},
    "lstm":         {"prob_up": 0.5297},
    "meta_learner": {"prob_up": 0.4457},
    "gemini":       {"prob_up": 0.55}
  },
  "sentiment": {
    "score": -0.267,
    "articles": [
      {
        "title": "...",
        "source": "Reuters",
        "url": "...",
        "published_at": "2026-04-22T14:00:00Z",
        "summary": "...",
        "impact_score": -0.8,
        "reasoning": "..."
      }
    ]
  },
  "composite": {
    "prob_up": 0.4781,
    "verdict": "bearish",
    "ai_overview": "The ensemble leans bearish for EURUSD..."
  },
  "disclaimer": "Educational use only. Not financial advice.",
  "stub": false
}
```

---

## Prediction Pipeline (per request)

```
POST /predict
  │
  ├─ asyncio.to_thread → predict_for_pair()
  │     └─ SQLite OHLC → make_features() → XGBoost → LSTM (48-bar window) → Meta-LR
  │
  ├─ fetch_news() — pair-specific AND query, 30-min cache
  │
  ├─ score_articles() — single Gemini batch call → impact_score per article
  │
  ├─ aggregate_sentiment() — confidence-weighted mean
  │
  ├─ asyncio.gather(
  │     generate_overview(),   ← Gemini narrative
  │     get_ai_prob()          ← Gemini directional probe
  │   )
  │
  └─ composite = 0.7 × meta_prob + 0.3 × ((sentiment + 1) / 2)
       verdict: >0.58 bullish, <0.42 bearish, else neutral
```

---

## Feature Engineering

All indicators implemented manually in `app/features/tabular.py`.

| Feature                    | Description                             |
| -------------------------- | --------------------------------------- |
| `log_return_1h`            | Log return over 1 bar                   |
| `log_return_3h/6h/12h/24h` | Multi-period log returns                |
| `vol_6h`, `vol_24h`        | Rolling std of 1h log returns           |
| `rsi_14`                   | RSI (Wilder EMA smoothing)              |
| `macd_line`, `macd_signal` | MACD(12,26,9)                           |
| `bbp_20`                   | Bollinger Band % (position within band) |
| `atr_14`                   | Average True Range                      |
| `adx_14`                   | Average Directional Index               |
| `stoch_k`                  | Stochastic %K(14,3)                     |
| `hour_sin`, `hour_cos`     | Cyclical encoding of hour-of-day        |
| `london_session`           | 1 if 07:00–15:59 UTC                    |
| `ny_session`               | 1 if 12:00–20:59 UTC                    |

**Target:** `1` if `close[t+1] > close[t]`, else `0`.

---

## Model Architecture

### XGBoost

- `objective="binary:logistic"`, `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=1.0`
- Early stopping (patience=30) on validation logloss

### LSTM

```
Input: (batch, 48, 5)   # 48-hour window × 5 channels
LSTM(5 → 32, batch_first=True)
Dropout(0.3)
Linear(32 → 1)
Sigmoid
Output: (batch,)        # prob_up
```

- Channels: open, high, low, close, log_return_1h
- Per-channel normalization using training set stats only
- Adam(lr=1e-3, weight_decay=1e-5), BCELoss, patience=5, max 30 epochs

### Meta-Learner

- `LogisticRegression(C=1.0, max_iter=500)`
- Input: `[xgb_val_prob, lstm_val_prob]`
- Trained on validation fold, evaluated on held-out test fold

**macOS note:** `train_meta.py` has zero torch imports. LSTM predictions are pre-saved as `.npy` files by `train_lstm.py` to avoid the xgboost + torch OpenMP deadlock on macOS. Serving code sets `KMP_DUPLICATE_LIB_OK=TRUE` before any imports.

---

## Clean Retrain from Scratch

```bash
rm app/data/forex.db
rm -rf models/EURUSD models/GBPUSD models/USDINR
# Re-run the full training pipeline from Step 1 above
```
