# ForexOracle — Technical Reference

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.14 |
| Package manager | uv |
| Database | SQLite (per-pair tables) |
| ML — tabular | XGBoost 3.x |
| ML — sequence | PyTorch 2.x LSTM |
| ML — ensemble | scikit-learn LogisticRegression |
| Backend | FastAPI + uvicorn |
| Validation | Pydantic v2 + pydantic-settings |
| News | NewsAPI v2 |
| Sentiment | Google Gemini (gemini-2.5-flash, batch scoring) |
| Frontend | React + TypeScript + Vite + Tailwind CSS v3.4 + Recharts |

---

## Repository Layout

```
.
├── app/
│   ├── ai/               # News fetch + Gemini sentiment pipeline
│   │   ├── news.py       # Async NewsAPI client, 30-min in-memory cache
│   │   ├── sentiment.py  # Gemini batch scorer (1 call per pair, not per article)
│   │   ├── aggregate.py  # Confidence-weighted sentiment aggregation
│   │   └── schemas.py    # RawArticle, ScoredArticle, GeminiBatchOutput
│   ├── api/
│   │   ├── main.py       # FastAPI app + CORS
│   │   └── routes/       # health, pairs, history, predict
│   ├── data/
│   │   ├── fetch.py      # yfinance → SQLite (2y hourly OHLC)
│   │   ├── split_data.py # Chronological train/test split → {pair}_TRAIN, {pair}_TEST
│   │   └── forex.db      # SQLite database (gitignored)
│   ├── evaluate/
│   │   └── evaluate_holdout.py  # XGBoost evaluation on held-out TEST split
│   ├── features/
│   │   └── tabular.py    # All 18 technical indicators (manual — no pandas-ta)
│   ├── models/
│   │   └── lstm_model.py # ForexLSTM nn.Module definition
│   ├── train/
│   │   ├── train_xgb.py  # XGBoost trainer
│   │   ├── train_lstm.py # LSTM trainer (also saves val/test .npy prediction files)
│   │   └── train_meta.py # LR meta-learner trainer (no torch import — avoids macOS OpenMP deadlock)
│   ├── config.py         # pydantic-settings: DB_PATH, API keys, SUPPORTED_PAIRS
│   └── db.py             # sqlite3 connection helper
├── frontend/
│   ├── src/
│   │   ├── components/   # PairPicker, VerdictCard, PredictionPanel
│   │   ├── lib/api.ts    # Typed API client (uses /api Vite proxy in dev)
│   │   └── App.tsx
│   └── vite.config.ts    # /api → http://localhost:8000 proxy
├── models/
│   └── {EURUSD,GBPUSD,USDINR}/
│       ├── xgb.json
│       ├── lstm.pt
│       ├── lstm_scaler.pkl        # {"mean": np.array(5), "std": np.array(5)}
│       ├── lstm_val_probs.npy     # Pre-computed val predictions (used by train_meta)
│       ├── lstm_val_labels.npy
│       ├── lstm_test_probs.npy
│       ├── lstm_test_labels.npy
│       ├── meta.pkl
│       ├── metrics.json           # xgboost + lstm + meta_learner + holdout keys
│       └── holdout_predictions.csv
├── scripts/
│   ├── smoke_test.py           # Loads all 3 models for EURUSD, prints 50-row prob table
│   └── test_news_pipeline.py   # End-to-end news + sentiment smoke test
├── .env                        # GEMINI_API_KEY, NEWS_API_KEY (not committed)
├── .env.example
└── pyproject.toml
```

---

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) installed
- Node.js 18+ (for frontend)
- A NewsAPI key (free tier: 100 req/day) — newsapi.org
- A Google Gemini API key (paid tier recommended) — aistudio.google.com

---

## First-Time Setup

```bash
# Install Python dependencies (creates .venv automatically)
uv sync

# Install the app package as editable (required for `app.*` imports)
uv pip install -e .

# Install frontend dependencies
cd frontend && npm install && cd ..

# Copy and fill in API keys
cp .env.example .env
# Edit .env: set GEMINI_API_KEY and NEWS_API_KEY
```

---

## Full Training Pipeline

Run these in order. Each step depends on the previous.

```bash
# 1. Fetch 2 years of hourly OHLC data into SQLite
uv run python -m app.data.fetch

# 2. Split into train (all but last 10 days) and test (last 10 days)
uv run python -m app.data.split_data

# 3. Train XGBoost — saves xgb.json + metrics.json per pair
uv run python -m app.train.train_xgb

# 4. Train LSTM — saves lstm.pt, lstm_scaler.pkl, and val/test .npy files per pair
#    Takes ~3–5 minutes on CPU. Set device to CUDA if available.
uv run python -m app.train.train_lstm

# 5. Train meta-learner — stacks XGBoost + LSTM val predictions via LogisticRegression
#    Saves meta.pkl, updates metrics.json
uv run python -m app.train.train_meta

# 6. (Optional) Evaluate XGBoost on the held-out 10-day test split
uv run python -m app.evaluate.evaluate_holdout
```

---

## Verification

```bash
# Confirm all 3 models load and produce predictions for EURUSD
uv run python scripts/smoke_test.py

# Confirm news fetch + Gemini sentiment pipeline works
uv run python scripts/test_news_pipeline.py
```

---

## Running the App

Terminal 1 — backend:
```bash
uv run uvicorn app.api.main:app --reload
# Listening on http://localhost:8000
```

Terminal 2 — frontend:
```bash
cd frontend && npm run dev
# Listening on http://localhost:5173
```

Open `http://localhost:5173` in a browser.

---

## API Reference

All endpoints are at `http://localhost:8000`. In dev the frontend proxies `/api/*` → `http://localhost:8000/*`.

### GET /health
```json
{"status": "ok"}
```

### GET /pairs
```json
{
  "pairs": [
    {"base": "EUR", "quote": "USD", "code": "EURUSD"},
    ...
  ]
}
```

### GET /history/{pair}?hours=168
Returns up to `hours` of hourly OHLC bars for the pair (max 1000h).
`pair` must be one of `EURUSD`, `GBPUSD`, `USDINR`.

```json
{
  "pair": "EURUSD",
  "bars": [
    {"datetime": "2026-04-22T10:00:00Z", "open": 1.134, "high": 1.137, "low": 1.133, "close": 1.135, "volume": 0},
    ...
  ]
}
```

### POST /predict
```json
// Request
{"base": "EUR", "quote": "USD"}

// Response
{
  "pair": "EURUSD",
  "current_price": 1.1352,
  "stub": true,
  "predictions": {
    "xgboost":     {"prob_up": 0.54, "prob_down": 0.46, "verdict": "bullish"},
    "lstm":        {"prob_up": 0.51, "prob_down": 0.49, "verdict": "neutral"},
    "meta_learner":{"prob_up": 0.53, "prob_down": 0.47, "verdict": "bullish"}
  },
  "sentiment": {"score": 0.0, "label": "neutral", "article_count": 0},
  "composite": {"prob_up": 0.52, "verdict": "bullish"},
  "ai_overview": "Stub prediction — models not yet wired in."
}
```

> **Note:** `stub: true` means the model probabilities are placeholder values. Real XGBoost + LSTM + meta inference is not yet wired into this endpoint.

---

## Feature Engineering

All indicators implemented manually in `app/features/tabular.py` (pandas-ta is incompatible with numpy 2.3+).

| Feature | Description |
|---------|-------------|
| `log_return_1h` | Log return over 1 bar |
| `log_return_3h/6h/12h/24h` | Multi-period log returns |
| `vol_6h`, `vol_24h` | Rolling std of 1h log returns |
| `rsi_14` | RSI (Wilder EMA smoothing) |
| `macd_line`, `macd_signal` | MACD(12,26,9) |
| `bbp_20` | Bollinger Band % (position within band) |
| `atr_14` | Average True Range |
| `adx_14` | Average Directional Index |
| `stoch_k` | Stochastic %K(14,3) |
| `hour_sin`, `hour_cos` | Cyclical encoding of hour-of-day |
| `london_session` | 1 if 07:00–15:59 UTC |
| `ny_session` | 1 if 12:00–20:59 UTC |

**Target:** `1` if `close[t+1] > close[t]`, else `0`. Approximately 50/50 split.

---

## Model Architecture

### XGBoost
- `objective="binary:logistic"`, `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`
- `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=1.0`
- Early stopping (patience=30) on validation logloss
- Input: 18 tabular features

### LSTM
```
Input: (batch, 48, 5)   # 48-hour window, 5 channels
LSTM(5 → 32, batch_first=True)
Dropout(0.3)
Linear(32 → 1)
Sigmoid → squeeze(-1)
Output: (batch,)        # probability of Up
```
- Channels: open, high, low, close, log_return_1h
- Normalized per-channel using training window stats only
- Adam(lr=1e-3, weight_decay=1e-5), BCELoss, early stopping patience=5, max 30 epochs

### Meta-Learner
- `LogisticRegression(C=1.0, max_iter=500)`
- Input: `[xgb_val_prob, lstm_val_prob]` (2 features)
- Trained on validation set predictions, evaluated on test set
- No torch import in this module (avoids macOS OpenMP deadlock when both torch and xgboost are loaded in the same process — LSTM predictions are pre-saved as `.npy` files by `train_lstm.py`)

---

## Known Issues

| Issue | Status |
|-------|--------|
| `/predict` returns stub values | Not yet wired — real model inference pending |
| macOS torch+xgboost OpenMP conflict | Worked around via pre-saved .npy files; serving code will need care |
| Frontend chart and metrics components | Placeholder boxes — not implemented |
| LSTM/meta not in holdout evaluation | evaluate_holdout.py only covers XGBoost |
| NewsAPI returns off-topic articles | Gemini scorer handles this correctly with low confidence scores |

---

## Clean Retrain from Scratch

```bash
rm app/data/forex.db
rm -rf models/EURUSD models/GBPUSD models/USDINR
# Then re-run the full training pipeline from Step 1 above
```
