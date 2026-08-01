"""
Thu thập tiêu đề tin tức Crypto từ CoinDesk, Cointelegraph, The Block.
Tương ứng mục 2.2.1 và Bước 1 (Chương 3) của báo cáo.

LƯU Ý: Cấu trúc HTML của các trang tin thay đổi theo thời gian, vì vậy các
selector CSS bên dưới có thể cần cập nhật lại khi trang thay đổi giao diện.
Với nhu cầu lấy dữ liệu lịch sử lớn, nên cân nhắc dùng dịch vụ archive
(vd. CryptoPanic API, NewsAPI) thay vì tự crawl.

Chạy:
    python src/data_collection/fetch_news.py --out data/raw/news_raw.csv --pages 50
"""
import argparse
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (research/education crawler)"}

SOURCES = {
    "coindesk": "https://www.coindesk.com/tag/bitcoin/{page}/",
    "cointelegraph": "https://cointelegraph.com/tags/bitcoin/page/{page}/",
    "theblock": "https://www.theblock.co/search?query=bitcoin&page={page}",
}


def fetch_page_titles(source: str, page: int) -> list:
    url = SOURCES[source].format(page=page)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [!] Lỗi tải {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    # Selector chung: lấy mọi thẻ <a> có heading bên trong (h2/h3) — cần tinh chỉnh
    # theo cấu trúc thực tế từng trang khi chạy thật.
    titles = []
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True)
        if text and len(text) > 15:
            titles.append(text)
    return titles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/news_raw.csv")
    parser.add_argument("--pages", type=int, default=50, help="Số trang crawl mỗi nguồn")
    args = parser.parse_args()

    records = []
    for source in SOURCES:
        print(f"Đang crawl nguồn: {source}")
        for page in range(1, args.pages + 1):
            titles = fetch_page_titles(source, page)
            now = pd.Timestamp.utcnow()
            for t in titles:
                records.append({"timestamp": now, "source": source, "title": t})
            time.sleep(1.0)  # lịch sự với server, tránh bị chặn

    df = pd.DataFrame(records).drop_duplicates(subset=["title"])
    df.to_csv(args.out, index=False)
    print(f"Đã lưu {len(df)} tiêu đề vào {args.out}")


if __name__ == "__main__":
    main()
