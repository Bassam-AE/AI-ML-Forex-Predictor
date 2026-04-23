import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from app.features.tabular import FEATURE_COLS, make_features, make_target

DB_PATH = "app/data/forex.db"
PAIRS = ["EURUSD", "GBPUSD", "USDINR"]


def load_ohlc(pair: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    table = f"{pair}_TRAIN" if f"{pair}_TRAIN" in tables else pair
    df = pd.read_sql(
        f'SELECT Datetime, Open, High, Low, Close, Volume FROM "{table}" ORDER BY Datetime',
        conn, parse_dates=["Datetime"], index_col="Datetime",
    )
    conn.close()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def train_one_pair(pair: str) -> dict | None:
    model_dir = Path(f"models/{pair}")
    lstm_val_path = model_dir / "lstm_val_probs.npy"
    lstm_test_path = model_dir / "lstm_test_probs.npy"

    if not lstm_val_path.exists():
        print(f"[WARN] {pair}: lstm_val_probs.npy missing — run train_lstm first, skipping")
        return None

    # --- Tabular data (same split as XGBoost) ---
    df = load_ohlc(pair)
    df = make_features(df)
    df = make_target(df)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    # --- XGBoost predictions ---
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(model_dir / "xgb.json")
    xgb_val_prob = xgb_model.predict_proba(val_df[FEATURE_COLS])[:, 1]
    xgb_test_prob = xgb_model.predict_proba(test_df[FEATURE_COLS])[:, 1]

    # --- LSTM predictions (pre-computed by train_lstm.py) ---
    lstm_val_prob = np.load(lstm_val_path)
    lstm_test_prob = np.load(lstm_test_path)

    # Align lengths — LSTM sequences start at SEQ_LEN-1 so may have slightly fewer rows
    min_val = min(len(xgb_val_prob), len(lstm_val_prob))
    min_test = min(len(xgb_test_prob), len(lstm_test_prob))

    X_meta_val = np.column_stack([xgb_val_prob[-min_val:], lstm_val_prob[-min_val:]])
    y_meta_val = val_df["target"].values[-min_val:]

    X_meta_test = np.column_stack([xgb_test_prob[-min_test:], lstm_test_prob[-min_test:]])
    y_meta_test = test_df["target"].values[-min_test:]

    # --- Train meta on val, evaluate on test ---
    meta = LogisticRegression(C=1.0, max_iter=500)
    meta.fit(X_meta_val, y_meta_val)

    meta_test_prob = meta.predict_proba(X_meta_test)[:, 1]
    meta_acc = accuracy_score(y_meta_test, (meta_test_prob > 0.5).astype(int))
    meta_ll = log_loss(y_meta_test, meta_test_prob)

    with open(model_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["meta_learner"] = {"accuracy": round(meta_acc, 6), "logloss": round(meta_ll, 6)}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    xgb_acc = accuracy_score(test_df["target"].values, (xgb_test_prob > 0.5).astype(int))
    lstm_acc = float(
        ((np.load(lstm_test_path) > 0.5).astype(int) ==
         np.load(model_dir / "lstm_test_labels.npy").astype(int)).mean()
    )

    return {"pair": pair, "xgb": xgb_acc, "lstm": lstm_acc, "meta": meta_acc}


if __name__ == "__main__":
    results = []
    for pair in PAIRS:
        try:
            r = train_one_pair(pair)
            if r:
                results.append(r)
        except Exception as exc:
            print(f"[ERROR] {pair}: {exc}")

    if results:
        print(f"\n{'Pair':<10} {'XGBoost':>8} {'LSTM':>8} {'Meta':>8}")
        print("-" * 38)
        for r in results:
            print(f"{r['pair']:<10} {r['xgb']:>8.4f} {r['lstm']:>8.4f} {r['meta']:>8.4f}")
