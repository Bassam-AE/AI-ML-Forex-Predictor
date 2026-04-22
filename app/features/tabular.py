import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers — all purely causal (no look-ahead)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = _ema(close, fast) - _ema(close, slow)
    sig = _ema(line, signal)
    return line, sig


def _bbp(close: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.Series:
    mid = close.rolling(length).mean()
    band = std_mult * close.rolling(length).std(ddof=0)
    return (close - (mid - band)) / (2 * band).replace(0, np.nan)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    atr = _atr(high, low, close, period)
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(com=period - 1, adjust=False).mean()


def _stoch_k(high: pd.Series, low: pd.Series, close: pd.Series,
             k: int = 14, smooth_k: int = 3) -> pd.Series:
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    return raw_k.rolling(smooth_k).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "log_return_1h",
    "log_return_3h",
    "log_return_6h",
    "log_return_12h",
    "log_return_24h",
    "vol_6h",
    "vol_24h",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "bbp_20",
    "atr_14",
    "adx_14",
    "stoch_k",
    "hour_sin",
    "hour_cos",
    "london_session",
    "ny_session",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]

    lr1 = np.log(close / close.shift(1))

    df["log_return_1h"] = lr1
    df["log_return_3h"] = np.log(close / close.shift(3))
    df["log_return_6h"] = np.log(close / close.shift(6))
    df["log_return_12h"] = np.log(close / close.shift(12))
    df["log_return_24h"] = np.log(close / close.shift(24))

    df["vol_6h"] = lr1.rolling(6).std(ddof=0)
    df["vol_24h"] = lr1.rolling(24).std(ddof=0)

    df["rsi_14"] = _rsi(close, 14)

    macd_line, macd_sig = _macd(close, 12, 26, 9)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_sig

    df["bbp_20"] = _bbp(close, 20, 2.0)
    df["atr_14"] = _atr(high, low, close, 14)
    df["adx_14"] = _adx(high, low, close, 14)
    df["stoch_k"] = _stoch_k(high, low, close, 14, 3)

    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["london_session"] = ((hour >= 7) & (hour < 16)).astype(int)
    df["ny_session"] = ((hour >= 12) & (hour < 21)).astype(int)

    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df
