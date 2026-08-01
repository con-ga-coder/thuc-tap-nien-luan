"""
Thu thập dữ liệu OHLCV BTC/USDT từ Binance qua endpoint /klines.
Tương ứng mục 2.2.1 và Bước 1 (Chương 3) của báo cáo.

Chạy:
    python src/data_collection/fetch_binance_ohlcv.py \
        --symbol BTCUSDT --interval 1h \
        --start 2024-01-01 --end 2026-07-01 \
        --out data/raw/ohlcv_raw.csv
"""
import argparse
import time
from datetime import datetime

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # giới hạn số nến mỗi lần gọi API của Binance


def date_to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Gọi Binance REST API theo từng lô (pagination) để lấy toàn bộ khoảng thời gian."""
    all_rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        cursor = rows[-1][6] + 1  # closeTime của nến cuối + 1ms
        time.sleep(0.3)  # tránh rate-limit
        print(f"  ... đã lấy {len(all_rows)} nến, tới {datetime.fromtimestamp(cursor/1000)}")
    return all_rows


def klines_to_dataframe(rows: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out", default="data/raw/ohlcv_raw.csv")
    args = parser.parse_args()

    start_ms = date_to_ms(args.start)
    end_ms = date_to_ms(args.end)

    print(f"Đang tải {args.symbol} [{args.interval}] từ {args.start} đến {args.end} ...")
    rows = fetch_klines(args.symbol, args.interval, start_ms, end_ms)
    df = klines_to_dataframe(rows)
    df.to_csv(args.out, index=False)
    print(f"Đã lưu {len(df)} bản ghi vào {args.out}")


if __name__ == "__main__":
    main()
