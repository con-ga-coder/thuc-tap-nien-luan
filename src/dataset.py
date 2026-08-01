"""
Chia dữ liệu theo thời gian (Chronological Split, KHÔNG random shuffle),
chuẩn hóa Min-Max (scaler chỉ fit trên tập Train để tránh Data Leakage),
và tạo cửa sổ trượt (sliding window, T=48) cho PyTorch Dataset.
Tương ứng mục 2.2.4 và Algorithm 1 (dòng 8-22) của báo cáo.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

TECH_COLS = [
    "open", "high", "low", "close", "volume",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "EMA7", "EMA14", "EMA21", "Volatility",
]  # 16 đặc trưng kỹ thuật = 5 OHLCV + 11 chỉ báo (không gồm sentiment)
SENT_COL = "hourly_sentiment"
TARGET_COL = "target"


def chronological_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """Chia dữ liệu tuyến tính theo trục thời gian, KHÔNG xáo trộn (mục 2.2.4)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Scaler CHỈ được fit trên tập Train (Algorithm 1, dòng 9) để tránh rò rỉ dữ liệu."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[TECH_COLS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    df = df.copy()
    df[TECH_COLS] = scaler.transform(df[TECH_COLS])
    return df


class TwoModalWindowDataset(Dataset):
    """
    Sinh mẫu (X_tech, X_sent, y) với cửa sổ trượt độ dài T=48.
    X_tech: (T, 16)  — chuỗi kỹ thuật đã chuẩn hóa
    X_sent: (T, 1)   — chuỗi sentiment score trong [-1, 1]
    y:      scalar   — nhãn nhị phân tại thời điểm cuối cửa sổ
    """

    def __init__(self, df_scaled: pd.DataFrame, lookback: int = 48):
        self.lookback = lookback
        self.tech = df_scaled[TECH_COLS].values.astype(np.float32)
        self.sent = df_scaled[[SENT_COL]].values.astype(np.float32)
        self.target = df_scaled[TARGET_COL].values.astype(np.int64)
        self.n_samples = len(df_scaled) - lookback

    def __len__(self):
        return max(self.n_samples, 0)

    def __getitem__(self, idx):
        start, end = idx, idx + self.lookback
        x_tech = torch.from_numpy(self.tech[start:end])          # (T, 16)
        x_sent = torch.from_numpy(self.sent[start:end])           # (T, 1)
        y = torch.tensor(self.target[end - 1], dtype=torch.long)  # nhãn tại cuối cửa sổ
        return x_tech, x_sent, y


def build_splits(csv_path: str, lookback: int = 48, train_ratio=0.70, val_ratio=0.15):
    """Hàm tiện ích: đọc CSV đã xử lý -> trả về 3 Dataset (train/val/test) + scaler đã fit."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    train_df, val_df, test_df = chronological_split(df, train_ratio, val_ratio)

    scaler = fit_scaler(train_df)
    train_scaled = apply_scaler(train_df, scaler)
    val_scaled = apply_scaler(val_df, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    train_ds = TwoModalWindowDataset(train_scaled, lookback)
    val_ds = TwoModalWindowDataset(val_scaled, lookback)
    test_ds = TwoModalWindowDataset(test_scaled, lookback)

    print(f"Train: {len(train_ds)} mẫu | Val: {len(val_ds)} mẫu | Test: {len(test_ds)} mẫu")
    return train_ds, val_ds, test_ds, scaler
