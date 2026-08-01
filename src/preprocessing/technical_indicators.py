"""
Tính toán 11 chỉ báo kỹ thuật phái sinh từ 5 trường OHLCV gốc.
Tương ứng mục 1.2.1 / 2.2.3 của báo cáo (16 đặc trưng kỹ thuật = 5 OHLCV + 11 chỉ báo).
"""
import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # 50 = trung tính khi chưa đủ dữ liệu


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def compute_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def compute_volatility(close: pd.Series, window: int = 14) -> pd.Series:
    """Độ biến động: độ lệch chuẩn của log-return trong cửa sổ trượt (mục 1.2.1)."""
    log_return = np.log(close / close.shift(1))
    return log_return.rolling(window).std()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame có cột [timestamp, open, high, low, close, volume]
    Output: DataFrame gốc + 11 cột chỉ báo phái sinh (16 đặc trưng kỹ thuật tổng cộng:
            5 OHLCV + 11 chỉ báo, khớp mục 1.2.1 / 2.2.3 của báo cáo).
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["RSI_14"] = compute_rsi(df["close"], period=14)

    macd, macd_signal, macd_hist = compute_macd(df["close"])
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(df["close"])
    df["BB_Upper"] = bb_upper
    df["BB_Middle"] = bb_mid
    df["BB_Lower"] = bb_lower

    df["EMA7"] = compute_ema(df["close"], 7)
    df["EMA14"] = compute_ema(df["close"], 14)
    df["EMA21"] = compute_ema(df["close"], 21)

    df["Volatility"] = compute_volatility(df["close"])

    # Các dòng đầu chuỗi (rolling window chưa đủ) sẽ có NaN -> backfill
    df = df.bfill()
    return df
