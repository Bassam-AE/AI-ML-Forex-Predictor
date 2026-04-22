import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from app.features.tabular import FEATURE_COLS, make_features, make_target

DB_PATH = "app/data/forex.db"
PAIRS = ["EURUSD", "GBPUSD", "USDINR"]


def load_ohlc(pair: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    # Prefer the pre-split training table if split_data.py has been run
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    table = f"{pair}_TRAIN" if f"{pair}_TRAIN" in tables else pair
    query = f'SELECT Datetime, Open, High, Low, Close, Volume FROM "{table}" ORDER BY Datetime'
    df = pd.read_sql(query, conn, parse_dates=["Datetime"], index_col="Datetime")
    conn.close()
    if table != pair:
        print(f"  Loading from {table} (holdout excluded)")
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df


def train_one_pair(pair: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {pair}")
    print(f"{'='*60}")

    df = load_ohlc(pair)
    df = make_features(df)
    df = make_target(df)

    needed = FEATURE_COLS + ["target"]
    df = df.dropna(subset=needed)

    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = df.iloc[:n_train]
    val = df.iloc[n_train: n_train + n_val]
    test = df.iloc[n_train + n_val:]

    X_train, y_train = train[FEATURE_COLS], train["target"]
    X_val, y_val = val[FEATURE_COLS], val["target"]
    X_test, y_test = test[FEATURE_COLS], test["target"]

    train_mean = y_train.mean()
    print(f"  Data shape   : {df.shape}")
    print(f"  Train / Val / Test : {len(train)} / {len(val)} / {len(test)}")
    print(f"  Train target mean  : {train_mean:.4f}", end="")
    if not (0.40 <= train_mean <= 0.60):
        print("  *** WARNING: target distribution outside [0.40, 0.60] ***", end="")
    print()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)
    xgb_acc = accuracy_score(y_test, pred)
    xgb_ll = log_loss(y_test, prob)

    acc_always_up = accuracy_score(y_test, np.ones(len(y_test), dtype=int))
    prev_dir = (X_test["log_return_1h"] > 0).astype(int)
    acc_prev_dir = accuracy_score(y_test, prev_dir)

    best_baseline = max(acc_always_up, acc_prev_dir)
    margin = xgb_acc - best_baseline
    verdict = "PASS" if margin >= 0.01 else "WEAK"

    print(f"  XGBoost accuracy   : {xgb_acc:.4f}", end="")
    if not (0.48 <= xgb_acc <= 0.58):
        print(f"  *** WARNING: accuracy outside expected range [0.48, 0.58] ***", end="")
    print()
    print(f"  XGBoost log-loss   : {xgb_ll:.4f}")
    print(f"  Baseline always-up : {acc_always_up:.4f}")
    print(f"  Baseline prev-dir  : {acc_prev_dir:.4f}")
    print(f"  Margin over best   : {margin:+.4f}  [{verdict}]")

    out_dir = Path(f"models/{pair}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "xgb.json")

    metrics = {
        "pair": pair,
        "test_samples": int(len(y_test)),
        "xgboost": {"accuracy": round(xgb_acc, 6), "logloss": round(xgb_ll, 6)},
        "baselines": {
            "always_up": {"accuracy": round(acc_always_up, 6)},
            "previous_direction": {"accuracy": round(acc_prev_dir, 6)},
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Artifacts saved to {out_dir}/")


if __name__ == "__main__":
    for pair in PAIRS:
        try:
            train_one_pair(pair)
        except Exception as exc:
            print(f"\n[ERROR] {pair}: {exc}")
