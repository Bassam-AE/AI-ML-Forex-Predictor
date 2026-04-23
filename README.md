# ForexOracle

An AI-powered forex direction predictor built as a capstone project. Combines a trained XGBoost + LSTM + meta-learner ensemble with live news sentiment analysis via Google Gemini to generate a directional probability and plain-English market overview for major currency pairs.

**Pairs supported:** EUR/USD · GBP/USD · USD/INR

---

## What it does

1. Runs three trained ML models (XGBoost, LSTM, Logistic Regression meta-learner) on the latest OHLC features to produce a probability that the base currency will rise in the next hour.
2. Fetches the last 72 hours of forex-relevant news via NewsAPI and uses Gemini to score each article's directional impact.
3. Asks Gemini directly for its own directional probability based on the news context.
4. Blends the ensemble model output (70%) and news sentiment (30%) into a single composite signal with a bullish / neutral / bearish verdict.
5. Generates a 2–3 sentence plain-English market overview via Gemini.

All results are served through a FastAPI backend and displayed in a React frontend with a live candlestick chart, model probability bars, news sidebar, and a sticky accuracy footer.

---

## Quick start

```bash
# 1. Clone and install Python dependencies
git clone <repo-url>
cd AI-ML-Forex-Predictor
uv sync && uv pip install -e .

# 2. Add API keys
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and NEWS_API_KEY

# 3. Install frontend dependencies
cd frontend && npm install && cd ..

# 4. Start backend (Terminal 1)
uv run uvicorn app.api.main:app --reload

# 5. Start frontend (Terminal 2)
cd frontend && npm run dev
```

Open `http://localhost:5173`, pick a pair, click **Analyze**.

The `models/` directory is committed — trained weights are included so the app runs immediately without retraining.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn, Python 3.14, uv |
| ML — tabular | XGBoost (300 trees, depth 4, early stopping) |
| ML — sequence | PyTorch LSTM (48-bar window, 5 channels → 32 hidden → 1) |
| ML — ensemble | scikit-learn LogisticRegression meta-learner |
| AI signals | Google Gemini 2.5 Flash (sentiment scoring + directional probe + overview) |
| News | NewsAPI v2 (pair-specific AND queries, 30-min cache) |
| Database | SQLite via yfinance (2 years hourly OHLC) |
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS v3 |
| Chart | TradingView lightweight-charts v5 |
| State | TanStack Query v5 |

---

## Training your own models

Trained weights are included, but if you want to retrain from scratch:

```bash
uv run python -m app.data.fetch          # fetch 2y hourly OHLC into SQLite
uv run python -m app.data.split_data     # chronological train/test split
uv run python -m app.train.train_xgb     # XGBoost
uv run python -m app.train.train_lstm    # LSTM (~3-5 min on CPU)
uv run python -m app.train.train_meta    # meta-learner
```

See [README_TECHNICAL.md](README_TECHNICAL.md) for the full pipeline, feature engineering details, and API reference.

---

## Disclaimer

Educational use only. This project does not constitute financial advice. Past model accuracy on a test split does not guarantee future performance.
