import os
import pickle
import sqlite3

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import xgboost as xgb  # import xgb before torch
import torch

from app.features.tabular import FEATURE_COLS, make_features, make_target
from app.models.lstm_model import ForexLSTM

PAIR = "EURUSD"
DB_PATH = "app/data/forex.db"
LSTM_CHANNELS = ["open", "high", "low", "close", "log_return_1h"]
SEQ_LEN = 48
N_ROWS = 50
# Need enough history for both indicator warmup (~26 bars) and LSTM window (SEQ_LEN-1=47 bars)
WARMUP = 100


def load_tail() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    table = f"{PAIR}_TRAIN" if f"{PAIR}_TRAIN" in tables else PAIR
    df = pd.read_sql(
        f'SELECT Datetime, Open, High, Low, Close, Volume FROM "{table}" ORDER BY Datetime',
        conn, parse_dates=["Datetime"], index_col="Datetime",
    )
    conn.close()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() for c in df.columns]
    return df.iloc[-(N_ROWS + WARMUP):]


# Load full warmup+display window, compute features on it all at once
df_full = load_tail()
df_full = make_features(df_full)
df_full = make_target(df_full)
df_full = df_full.dropna(subset=FEATURE_COLS + ["target"])

# Display slice (last N_ROWS only)
df = df_full.iloc[-N_ROWS:]

# XGBoost — only needs the display slice
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(f"models/{PAIR}/xgb.json")
xgb_probs = xgb_model.predict_proba(df[FEATURE_COLS])[:, 1]

# LSTM — build windows from the full df, record probs indexed to display rows
with open(f"models/{PAIR}/lstm_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

lstm_model = ForexLSTM(n_features=5)
lstm_model.load_state_dict(
    torch.load(f"models/{PAIR}/lstm.pt", map_location="cpu", weights_only=True)
)
lstm_model.eval()

full_vals = df_full[LSTM_CHANNELS].values.astype(np.float32)
mean, std = scaler["mean"], scaler["std"]

# For each row in the display slice, find its position in df_full and build window
display_start_in_full = len(df_full) - N_ROWS
lstm_probs = []
for i in range(N_ROWS):
    pos = display_start_in_full + i
    if pos < SEQ_LEN - 1:
        lstm_probs.append(float("nan"))
        continue
    window = full_vals[pos - SEQ_LEN + 1 : pos + 1]
    w_norm = (window - mean) / std
    t = torch.from_numpy(w_norm).unsqueeze(0)
    with torch.no_grad():
        lstm_probs.append(lstm_model(t).item())

lstm_probs = np.array(lstm_probs)

# Meta
with open(f"models/{PAIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

meta_input = np.column_stack([xgb_probs, lstm_probs])
valid_mask = ~np.isnan(meta_input).any(axis=1)
meta_probs = np.full(N_ROWS, float("nan"))
meta_probs[valid_mask] = meta.predict_proba(meta_input[valid_mask])[:, 1]

# Print table
print(f"\n{'Timestamp':<25} {'XGB':>6} {'LSTM':>6} {'Meta':>6}")
print("-" * 48)
for i, (ts, _) in enumerate(df.iterrows()):
    xp = f"{xgb_probs[i]:.3f}"
    lp = f"{lstm_probs[i]:.3f}" if not np.isnan(lstm_probs[i]) else "  n/a"
    mp = f"{meta_probs[i]:.3f}" if not np.isnan(meta_probs[i]) else "  n/a"
    print(f"{str(ts)[:25]:<25} {xp:>6} {lp:>6} {mp:>6}")
