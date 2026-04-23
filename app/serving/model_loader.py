import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb  # must import before torch on macOS
import torch

from app.config import DB_PATH, SUPPORTED_PAIRS
from app.features.tabular import FEATURE_COLS, make_features
from app.models.lstm_model import ForexLSTM

LSTM_CHANNELS = ["open", "high", "low", "close", "log_return_1h"]
SEQ_LEN = 48
WARMUP = 100


def load_pair_models(pair: str) -> dict:
    model_dir = Path(f"models/{pair}")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_dir / "xgb.json")

    has_lstm = (model_dir / "lstm.pt").exists()
    lstm_model = None
    lstm_scaler = None
    if has_lstm:
        with open(model_dir / "lstm_scaler.pkl", "rb") as f:
            lstm_scaler = pickle.load(f)
        lstm_model = ForexLSTM(n_features=5)
        lstm_model.load_state_dict(
            torch.load(model_dir / "lstm.pt", map_location="cpu", weights_only=True)
        )
        lstm_model.eval()

    has_meta = (model_dir / "meta.pkl").exists()
    meta_model = None
    if has_meta:
        with open(model_dir / "meta.pkl", "rb") as f:
            meta_model = pickle.load(f)

    return {
        "xgb": xgb_model,
        "lstm": lstm_model,
        "lstm_scaler": lstm_scaler,
        "meta": meta_model,
        "has_lstm": has_lstm,
        "has_meta": has_meta,
    }


def load_all_models() -> dict[str, dict]:
    bundles: dict[str, dict] = {}
    pairs = [p["pair"] for p in SUPPORTED_PAIRS]
    for pair in pairs:
        try:
            bundles[pair] = load_pair_models(pair)
            print(f"[model_loader] loaded models for {pair}")
        except Exception as exc:
            print(f"[model_loader] WARN: could not load {pair}: {exc}")
    print(f"[model_loader] Models loaded for {len(bundles)} pairs")
    return bundles


def _load_ohlc(pair: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    table = f"{pair}_TRAIN" if f"{pair}_TRAIN" in tables else pair
    df = pd.read_sql(
        f'SELECT Datetime, Open, High, Low, Close, Volume FROM "{table}" '
        f'ORDER BY Datetime DESC LIMIT {WARMUP + SEQ_LEN + 50}',
        conn, parse_dates=["Datetime"], index_col="Datetime",
    )
    conn.close()
    df = df.sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def predict_for_pair(pair: str, bundle: dict) -> dict:
    df = _load_ohlc(pair)
    df = make_features(df)
    df = df.dropna(subset=FEATURE_COLS)

    last_row = df.iloc[[-1]]
    xgb_prob = float(bundle["xgb"].predict_proba(last_row[FEATURE_COLS])[0, 1])

    lstm_prob = xgb_prob
    if bundle["has_lstm"]:
        scaler = bundle["lstm_scaler"]
        vals = df[LSTM_CHANNELS].values.astype(np.float32)
        if len(vals) >= SEQ_LEN:
            window = vals[-SEQ_LEN:]
            w_norm = (window - scaler["mean"]) / scaler["std"]
            t = torch.from_numpy(w_norm).unsqueeze(0)
            with torch.no_grad():
                lstm_prob = float(bundle["lstm"](t).item())

    meta_prob = xgb_prob
    if bundle["has_meta"]:
        meta_input = np.array([[xgb_prob, lstm_prob]])
        meta_prob = float(bundle["meta"].predict_proba(meta_input)[0, 1])

    current_price = float(df["close"].iloc[-1])
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "xgb_prob": xgb_prob,
        "lstm_prob": lstm_prob,
        "meta_prob": meta_prob,
        "current_price": current_price,
        "timestamp": timestamp,
    }
