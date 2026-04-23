#!/usr/bin/env python3
"""
evaluate_10h.py
Roll-forward 10-step evaluation: XGBoost, LSTM, and Meta-Learner.

For each of the 3 pairs, predicts the next 10 hours one step at a time
using only data available at that moment (no look-ahead), then compares
predicted price paths to the actual held-out prices.

Run from repo root:
    uv run python evaluate_10h.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import sqlite3
from pathlib import Path

import xgboost as xgb          # must import before torch on macOS
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from app.features.tabular import FEATURE_COLS, make_features
from app.models.lstm_model import ForexLSTM

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH  = "app/data/forex.db"
PAIRS    = ["EURUSD", "GBPUSD", "USDINR"]
N_HOURS  = 10
CTX_ROWS = 200   # training-tail rows for indicator warm-up
SEQ_LEN  = 48    # LSTM look-back window
LSTM_CHAN = ["open", "high", "low", "close", "log_return_1h"]

COLORS = {
    "actual": "#111111",
    "xgb":    "#2196F3",
    "lstm":   "#FF9800",
    "meta":   "#9C27B0",
    "ok":     "#4CAF50",
    "wrong":  "#F44336",
}


# ── Data helpers ──────────────────────────────────────────────────────────────
def _read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql(
        f'SELECT Datetime, Open, High, Low, Close FROM "{table}"',
        conn, parse_dates=["Datetime"], index_col="Datetime",
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(pair: str) -> tuple:
    d = Path(f"models/{pair}")

    xgb_m = xgb.XGBClassifier()
    xgb_m.load_model(d / "xgb.json")

    with open(d / "lstm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    lstm_m = ForexLSTM(n_features=5)
    lstm_m.load_state_dict(
        torch.load(d / "lstm.pt", map_location="cpu", weights_only=True)
    )
    lstm_m.eval()

    with open(d / "meta.pkl", "rb") as f:
        meta_m = pickle.load(f)

    return xgb_m, lstm_m, scaler, meta_m


# ── Single-step inference ─────────────────────────────────────────────────────
def predict_step(
    ctx: pd.DataFrame,
    xgb_m, lstm_m, scaler, meta_m,
) -> tuple[float, float, float]:
    df = make_features(ctx.copy()).dropna(subset=FEATURE_COLS)
    if df.empty:
        return 0.5, 0.5, 0.5

    last = df.iloc[[-1]]
    xgb_p = float(xgb_m.predict_proba(last[FEATURE_COLS])[0, 1])

    vals = df[LSTM_CHAN].values.astype(np.float32)
    if len(vals) >= SEQ_LEN:
        win = (vals[-SEQ_LEN:] - scaler["mean"]) / scaler["std"]
        t = torch.from_numpy(win).unsqueeze(0)
        with torch.no_grad():
            lstm_p = float(lstm_m(t).item())
    else:
        lstm_p = xgb_p

    meta_p = float(meta_m.predict_proba(np.array([[xgb_p, lstm_p]]))[0, 1])
    return xgb_p, lstm_p, meta_p


# ── Roll-forward evaluation for one pair ─────────────────────────────────────
def evaluate_pair(pair: str, conn: sqlite3.Connection) -> pd.DataFrame:
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    train_tbl  = f"{pair}_TRAIN" if f"{pair}_TRAIN" in tables else pair
    train_tail = _read_table(conn, train_tbl).iloc[-CTX_ROWS:]
    # Need N+1 rows so we have the "next close" for every prediction
    test_all   = _read_table(conn, f"{pair}_TEST").iloc[: N_HOURS + 1]

    if len(test_all) < N_HOURS + 1:
        raise RuntimeError(
            f"{pair}_TEST has only {len(test_all)} rows — need at least {N_HOURS + 1}."
        )

    xgb_m, lstm_m, scaler, meta_m = load_models(pair)

    rows = []
    for i in range(N_HOURS):
        # Context = train tail + test bars 0..i (inclusive — no look-ahead on bar i+1)
        ctx = pd.concat([train_tail, test_all.iloc[: i + 1]])
        xp, lp, mp = predict_step(ctx, xgb_m, lstm_m, scaler, meta_m)

        close_now  = float(test_all["close"].iloc[i])
        close_next = float(test_all["close"].iloc[i + 1])

        rows.append(
            dict(
                timestamp  = test_all.index[i],
                close      = close_now,
                close_next = close_next,
                actual_up  = int(close_next > close_now),
                xgb_prob   = xp,
                lstm_prob  = lp,
                meta_prob  = mp,
            )
        )

    return pd.DataFrame(rows)


# ── Simulated price path from probability forecasts ───────────────────────────
def make_price_path(start: float, probs: np.ndarray, avg_abs_ret: float) -> np.ndarray:
    """
    Translate a sequence of P(up) forecasts into a price path.
    Confidence is linearly scaled: prob=0.5 → no move, prob=1.0 → +avg_abs_ret.
    Returns N+1 prices (start + one per forecast).
    """
    path = [start]
    for p in probs:
        confidence = (p - 0.5) * 2          # maps [0,1] → [-1, +1]
        path.append(path[-1] * np.exp(confidence * avg_abs_ret))
    return np.array(path)


# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(pair: str, df: pd.DataFrame) -> None:
    print(f"\n  {pair}  —  {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M UTC')}"
          f"  →  {df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')}")
    for prob_key, label in [
        ("xgb_prob",  "XGBoost     "),
        ("lstm_prob", "LSTM        "),
        ("meta_prob", "Meta-Learner"),
    ]:
        preds = (df[prob_key] >= 0.5).astype(int).values
        acc   = accuracy_score(df["actual_up"], preds)
        n_ok  = int(round(acc * N_HOURS))
        dirs  = ["↑" if p else "↓" for p in preds]
        print(f"    {label}  {acc:.0%}  ({n_ok}/{N_HOURS})  {' '.join(dirs)}")
    actual_dirs = ["↑" if u else "↓" for u in df["actual_up"]]
    print(f"    {'Actual      '}  {'—':>3}  {'—':>5}  {' '.join(actual_dirs)}")


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_results(all_results: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(22, 13))
    fig.suptitle(
        "10-Hour Roll-Forward Evaluation  |  Predicted Price Paths vs Actual  "
        "|  Dot colours = Meta-Learner correct (green) / wrong (red)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[2.2, 1],
        hspace=0.52, wspace=0.36,
    )

    for cidx, pair in enumerate(PAIRS):
        df = all_results[pair]

        x_prob  = np.arange(N_HOURS)           # 0..9  (probability panel)
        x_price = np.arange(N_HOURS + 1)       # 0..10 (price panel, inclusive endpoint)

        # Full actual price line: close[0] … close[N]
        actual_prices = np.append(df["close"].values, df["close_next"].iloc[-1])

        # Average absolute log-return for price-path scaling
        log_rets = np.log(actual_prices[1:] / actual_prices[:-1])
        avg_ret  = float(np.abs(log_rets).mean()) or 0.0005

        start     = actual_prices[0]
        xgb_path  = make_price_path(start, df["xgb_prob"].values,  avg_ret)
        lstm_path = make_price_path(start, df["lstm_prob"].values,  avg_ret)
        meta_path = make_price_path(start, df["meta_prob"].values,  avg_ret)

        # Tick labels: N+1 timestamps for price chart
        ts_labels = [t.strftime("%d %b\n%H:%M") for t in df["timestamp"]]
        last_ts   = (df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).strftime("%d %b\n%H:%M")
        ts_price  = ts_labels + [last_ts]

        # ── Price panel ───────────────────────────────────────────────────
        ax_p = fig.add_subplot(gs[0, cidx])

        ax_p.plot(x_price, actual_prices, color=COLORS["actual"], lw=2.5,
                  marker="o", markersize=5, label="Actual", zorder=5)
        ax_p.plot(x_price, xgb_path,  color=COLORS["xgb"],  lw=1.6, ls="--",
                  label="XGBoost",      alpha=0.85)
        ax_p.plot(x_price, lstm_path, color=COLORS["lstm"], lw=1.6, ls="--",
                  label="LSTM",         alpha=0.85)
        ax_p.plot(x_price, meta_path, color=COLORS["meta"], lw=2.2, ls="-.",
                  label="Meta-Learner", alpha=0.90)

        # Colour-coded correctness dots on actual line (using Meta-Learner)
        meta_preds = (df["meta_prob"].values >= 0.5).astype(int)
        for i in range(N_HOURS):
            correct = meta_preds[i] == df["actual_up"].iloc[i]
            ax_p.scatter(
                i, actual_prices[i],
                color=COLORS["ok"] if correct else COLORS["wrong"],
                s=90, zorder=8, edgecolors="white", linewidths=1.2,
            )

        # Accuracy box
        accs = {}
        for key, lbl in [("xgb_prob", "XGB"), ("lstm_prob", "LSTM"), ("meta_prob", "Meta")]:
            accs[lbl] = accuracy_score(
                df["actual_up"], (df[key] >= 0.5).astype(int)
            )
        acc_str = "  ".join(f"{k}: {v:.0%}" for k, v in accs.items())
        ax_p.text(
            0.01, 0.02, f"Accuracy  —  {acc_str}",
            transform=ax_p.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88),
        )

        ax_p.set_title(pair, fontsize=14, fontweight="bold")
        ax_p.set_ylabel("Price", fontsize=9)
        ax_p.set_xticks(x_price)
        ax_p.set_xticklabels(ts_price, fontsize=6.3)
        ax_p.legend(fontsize=7.5, loc="upper left")
        ax_p.grid(alpha=0.28)

        # ── Probability panel ─────────────────────────────────────────────
        ax_q = fig.add_subplot(gs[1, cidx])

        ax_q.plot(x_prob, df["xgb_prob"],  color=COLORS["xgb"],  lw=1.6,
                  marker="o", ms=5, label="XGBoost")
        ax_q.plot(x_prob, df["lstm_prob"], color=COLORS["lstm"], lw=1.6,
                  marker="s", ms=5, label="LSTM")
        ax_q.plot(x_prob, df["meta_prob"], color=COLORS["meta"], lw=2.2,
                  marker="D", ms=5, label="Meta-Learner")
        ax_q.axhline(0.5, color="gray", ls=":", lw=1.5, label="0.5 threshold")

        # Background shading = actual price direction
        for i in range(N_HOURS):
            shade = COLORS["ok"] if df["actual_up"].iloc[i] else COLORS["wrong"]
            ax_q.axvspan(i - 0.45, i + 0.45, alpha=0.10, color=shade)

        up_patch   = mpatches.Patch(color=COLORS["ok"],    alpha=0.35, label="Actual ↑")
        down_patch = mpatches.Patch(color=COLORS["wrong"], alpha=0.35, label="Actual ↓")
        handles, labels_ = ax_q.get_legend_handles_labels()
        ax_q.legend(
            handles + [up_patch, down_patch],
            labels_ + ["Actual ↑", "Actual ↓"],
            fontsize=6.5, loc="upper right", ncol=2,
        )

        ax_q.set_ylim(0.0, 1.0)
        ax_q.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_q.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=7)
        ax_q.set_ylabel("P(↑)", fontsize=9)
        ax_q.set_xticks(x_prob)
        ax_q.set_xticklabels(ts_labels, fontsize=6.3)
        ax_q.set_xlabel("Hour (UTC)", fontsize=8)
        ax_q.grid(alpha=0.28)

    plt.savefig("evaluation_10_hours.png", dpi=150, bbox_inches="tight")
    print("\nSaved → evaluation_10_hours.png")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 10-Hour Roll-Forward Evaluation ===")
    conn = sqlite3.connect(DB_PATH)
    all_results: dict[str, pd.DataFrame] = {}

    for pair in PAIRS:
        print(f"\n[{pair}] running...")
        try:
            df = evaluate_pair(pair, conn)
            all_results[pair] = df
            print_summary(pair, df)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            raise

    conn.close()

    if all_results:
        print("\nGenerating chart...")
        plot_results(all_results)
