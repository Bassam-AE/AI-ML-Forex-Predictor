import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from app.features.tabular import FEATURE_COLS, make_features, make_target

DB_PATH = "app/data/forex.db"
PAIRS = ["EURUSD", "GBPUSD", "USDINR"]
# Must cover the longest indicator lookback (MACD slow=26 + signal=9 + ADX smoothing)
CONTEXT_ROWS = 60


def load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql(f'SELECT * FROM "{table}"', conn, parse_dates=["Datetime"], index_col="Datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def evaluate_pair(pair: str) -> dict:
    model_path = Path(f"models/{pair}/xgb.json")
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path} — run train_xgb.py first.")

    conn = sqlite3.connect(DB_PATH)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

    if f"{pair}_TEST" not in tables:
        conn.close()
        raise RuntimeError(f"{pair}_TEST table not found — run split_data.py first.")

    test_raw = load_table(conn, f"{pair}_TEST")

    train_table = f"{pair}_TRAIN" if f"{pair}_TRAIN" in tables else pair
    train_tail = load_table(conn, train_table).iloc[-CONTEXT_ROWS:]
    conn.close()

    # Combine tail of training data with test data for indicator warmup
    combined = pd.concat([train_tail, test_raw])
    combined = make_features(combined)
    combined = make_target(combined)

    # Slice back to only the test window
    test = combined.loc[test_raw.index]
    test = test.dropna(subset=FEATURE_COLS + ["target"])

    X_test = test[FEATURE_COLS]
    y_test = test["target"]

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob > 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, prob)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=["Down", "Up"], output_dict=True, zero_division=0)

    # Direction breakdown
    n_up_actual = int(y_test.sum())
    n_down_actual = int((y_test == 0).sum())
    n_up_pred = int(pred.sum())
    n_down_pred = int((pred == 0).sum())

    # Date range of the holdout
    date_start = test.index.min().strftime("%Y-%m-%d %H:%M UTC")
    date_end = test.index.max().strftime("%Y-%m-%d %H:%M UTC")

    return {
        "pair": pair,
        "holdout_rows": len(test),
        "date_range": f"{date_start}  →  {date_end}",
        "accuracy": acc,
        "logloss": ll,
        "confusion_matrix": cm,
        "report": report,
        "actual": {"up": n_up_actual, "down": n_down_actual},
        "predicted": {"up": n_up_pred, "down": n_down_pred},
        "prob": prob,
        "y_test": y_test,
        "pred": pred,
        "index": test.index,
    }


def print_results(r: dict) -> None:
    pair = r["pair"]
    print(f"\n{'='*62}")
    print(f"  {pair}  —  Holdout Evaluation")
    print(f"{'='*62}")
    print(f"  Period      : {r['date_range']}")
    print(f"  Rows        : {r['holdout_rows']} hourly bars")
    print()
    print(f"  Accuracy    : {r['accuracy']:.4f}  ({r['accuracy']*100:.1f}%)")
    print(f"  Log-loss    : {r['logloss']:.4f}")
    print()
    print(f"  Actual moves    — Up: {r['actual']['up']:>4}   Down: {r['actual']['down']:>4}")
    print(f"  Predicted moves — Up: {r['predicted']['up']:>4}   Down: {r['predicted']['down']:>4}")
    print()

    cm = r["confusion_matrix"]
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred Down   Pred Up")
    print(f"  Actual Down  {cm[0,0]:>9}   {cm[0,1]:>7}")
    print(f"  Actual Up    {cm[1,0]:>9}   {cm[1,1]:>7}")
    print()

    rpt = r["report"]
    print("  Per-class performance:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for label in ["Down", "Up"]:
        m = rpt[label]
        print(f"  {label:<10} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1-score']:>8.4f}")

    # Simple verdict
    baseline_acc = max(r["actual"]["up"], r["actual"]["down"]) / r["holdout_rows"]
    margin = r["accuracy"] - baseline_acc
    verdict = "PASS" if margin >= 0.01 else "WEAK"
    print()
    print(f"  Majority-class baseline : {baseline_acc:.4f}")
    print(f"  Margin over baseline    : {margin:+.4f}  [{verdict}]")


def save_results(r: dict) -> None:
    out_dir = Path(f"models/{r['pair']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Append holdout results into metrics.json
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["holdout"] = {
        "rows": r["holdout_rows"],
        "date_range": r["date_range"],
        "accuracy": round(r["accuracy"], 6),
        "logloss": round(r["logloss"], 6),
        "actual": r["actual"],
        "predicted": r["predicted"],
        "per_class": {
            "down": {k: round(v, 6) for k, v in r["report"]["Down"].items() if k != "support"},
            "up": {k: round(v, 6) for k, v in r["report"]["Up"].items() if k != "support"},
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Prediction log — timestamp, actual, predicted, probability
    log = pd.DataFrame({
        "ts_utc": r["index"],
        "actual": r["y_test"].values,
        "predicted": r["pred"],
        "prob_up": r["prob"].round(4),
        "correct": (r["y_test"].values == r["pred"]).astype(int),
    })
    log.to_csv(out_dir / "holdout_predictions.csv", index=False)
    print(f"\n  Results saved to {out_dir}/metrics.json and holdout_predictions.csv")


if __name__ == "__main__":
    for pair in PAIRS:
        try:
            results = evaluate_pair(pair)
            print_results(results)
            save_results(results)
        except Exception as exc:
            print(f"\n[ERROR] {pair}: {exc}")
