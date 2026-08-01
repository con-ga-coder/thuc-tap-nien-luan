"""
Xây dựng bộ dữ liệu đặc trưng cuối cùng (17 features + target) từ dữ liệu thô.
Tương ứng Algorithm 1 (dòng 3-7) và mục 2.2.2 / 2.2.3 của báo cáo:
  1. Nội suy tuyến tính bù giá trị khuyết thiếu (Linear Interpolation)
  2. Loại bỏ outlier bằng IQR (clamping)
  3. Tính 11 chỉ báo kỹ thuật phái sinh -> 16 đặc trưng kỹ thuật
  4. Tính sentiment score cho từng tiêu đề, gộp theo giờ (Hourly Aggregation)
  5. Đồng bộ thời gian (Alignment) giữa 2 luồng dữ liệu
  6. Gán nhãn xu hướng giá sau 24h (Target Labeling)

Chạy:
    python src/preprocessing/build_dataset.py \
        --ohlcv data/raw/ohlcv_raw.csv \
        --news data/raw/news_raw.csv \
        --out data/processed/dataset_features.csv
"""
import argparse

import numpy as np
import pandas as pd

from src.preprocessing.technical_indicators import add_technical_indicators
from src.preprocessing.sentiment_analysis import score_news_dataframe

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "EMA7", "EMA14", "EMA21", "Volatility",
    "hourly_sentiment",
]


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Bước 1-2: nội suy tuyến tính cho khung giờ khuyết + clamp outlier bằng IQR."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Tạo trục thời gian đầy đủ theo giờ để phát hiện khung giờ bị thiếu
    full_range = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1h")
    df = df.set_index("timestamp").reindex(full_range)
    df.index.name = "timestamp"

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()

    # Loại bỏ outlier bằng IQR (clamping thay vì xóa dòng)
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df.reset_index()


def aggregate_sentiment_hourly(news_df: pd.DataFrame, ohlcv_timestamps: pd.Series) -> pd.DataFrame:
    """Bước 4-5: gộp sentiment theo giờ, khung giờ không có tin -> 0.0 (trung lập)."""
    news_df = news_df.copy()
    news_df["timestamp"] = pd.to_datetime(news_df["timestamp"])
    news_df["hour"] = news_df["timestamp"].dt.floor("h")

    hourly = news_df.groupby("hour")["sentiment_score"].mean().rename("hourly_sentiment")

    frame = pd.DataFrame({"timestamp": ohlcv_timestamps})
    frame["hour"] = frame["timestamp"].dt.floor("h")
    frame = frame.merge(hourly, left_on="hour", right_index=True, how="left")
    frame["hourly_sentiment"] = frame["hourly_sentiment"].fillna(0.0)  # trung lập, không phải "không biết"
    return frame[["timestamp", "hourly_sentiment"]]


def label_target(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """Bước 6: y = 1 nếu Close(t+24h) > Close(t), ngược lại y = 0."""
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon_hours)
    df["target"] = (df["future_close"] > df["close"]).astype(int)
    df = df.dropna(subset=["future_close"]).drop(columns=["future_close"])
    return df


def build_dataset(ohlcv_path: str, news_path: str, out_path: str) -> pd.DataFrame:
    print("[1/5] Đang làm sạch OHLCV (interpolation + IQR clamping)...")
    ohlcv_raw = pd.read_csv(ohlcv_path)
    ohlcv_clean = clean_ohlcv(ohlcv_raw)

    print("[2/5] Đang tính 11 chỉ báo kỹ thuật phái sinh...")
    ohlcv_feat = add_technical_indicators(ohlcv_clean)

    print("[3/5] Đang chạy sentiment analysis (DistilBERT) trên tiêu đề tin tức...")
    news_raw = pd.read_csv(news_path)
    news_scored = score_news_dataframe(news_raw)

    print("[4/5] Đang gộp sentiment theo giờ và đồng bộ thời gian...")
    sentiment_hourly = aggregate_sentiment_hourly(news_scored, ohlcv_feat["timestamp"])
    merged = ohlcv_feat.merge(sentiment_hourly, on="timestamp", how="left")
    merged["hourly_sentiment"] = merged["hourly_sentiment"].fillna(0.0)

    print("[5/5] Đang gán nhãn xu hướng giá 24h tiếp theo...")
    final_df = label_target(merged, horizon_hours=24)

    final_df = final_df[["timestamp"] + FEATURE_COLS + ["target"]]
    final_df.to_csv(out_path, index=False)
    print(f"Đã lưu dataset cuối cùng: {len(final_df)} bản ghi, {len(FEATURE_COLS)} đặc trưng -> {out_path}")
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="data/raw/ohlcv_raw.csv")
    parser.add_argument("--news", default="data/raw/news_raw.csv")
    parser.add_argument("--out", default="data/processed/dataset_features.csv")
    args = parser.parse_args()
    build_dataset(args.ohlcv, args.news, args.out)


if __name__ == "__main__":
    main()
