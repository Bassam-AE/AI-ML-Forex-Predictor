import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.features.tabular import make_features, make_target
from app.models.lstm_model import ForexLSTM

torch.manual_seed(42)
np.random.seed(42)

DB_PATH = "app/data/forex.db"
PAIRS = ["EURUSD", "GBPUSD", "USDINR"]
LSTM_CHANNELS = ["open", "high", "low", "close", "log_return_1h"]
SEQ_LEN = 48
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5


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


def build_sequences(df: pd.DataFrame):
    drop_cols = ["open", "high", "low", "close", "log_return_1h", "target"]
    clean = df.dropna(subset=drop_cols)
    vals = clean[LSTM_CHANNELS].values.astype(np.float32)
    labels = clean["target"].values.astype(np.float32)

    X, y = [], []
    for t in range(SEQ_LEN - 1, len(clean)):
        window = vals[t - SEQ_LEN + 1 : t + 1]
        if np.isnan(window).any():
            continue
        X.append(window)
        y.append(labels[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def chronological_split(n: int):
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return n_train, n_val


def train_one_pair(pair: str, device: torch.device) -> None:
    df = load_ohlc(pair)
    df = make_features(df)
    df = make_target(df)

    X, y = build_sequences(df)
    n = len(X)
    n_train, n_val = chronological_split(n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

    # Normalize per channel using training stats only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    scaler = {"mean": mean.squeeze(), "std": std.squeeze()}
    out_dir = Path(f"models/{pair}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lstm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=BATCH_SIZE,
    )
    test_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE,
    )

    model = ForexLSTM(n_features=5).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs = model(xb.to(device)).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.numpy())

    test_probs = np.concatenate(all_probs)
    test_labels = np.concatenate(all_labels)
    preds = (test_probs > 0.5).astype(int)
    accuracy = float((preds == test_labels).mean())

    eps = 1e-7
    p = np.clip(test_probs, eps, 1 - eps)
    logloss = float(-np.mean(test_labels * np.log(p) + (1 - test_labels) * np.log(1 - p)))

    # Generate and save val probs so train_meta.py never needs to import torch
    val_probs_list = []
    with torch.no_grad():
        for xb, _ in val_dl:
            val_probs_list.append(model(xb.to(device)).cpu().numpy())
    val_probs = np.concatenate(val_probs_list)

    np.save(out_dir / "lstm_val_probs.npy", val_probs)
    np.save(out_dir / "lstm_val_labels.npy", y_val)
    np.save(out_dir / "lstm_test_probs.npy", test_probs)
    np.save(out_dir / "lstm_test_labels.npy", y_test)

    torch.save(model.state_dict(), out_dir / "lstm.pt")

    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["lstm"] = {"accuracy": round(accuracy, 6), "logloss": round(logloss, 6)}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"{pair}: LSTM acc={accuracy:.4f} logloss={logloss:.4f}  "
          f"[train={n_train} val={n_val} test={len(X_test)}]")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    for pair in PAIRS:
        try:
            train_one_pair(pair, device)
        except Exception as exc:
            print(f"[ERROR] {pair}: {exc}")
