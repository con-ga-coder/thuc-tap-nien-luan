# Dự báo xu hướng giá BTC/USDT 24h dựa trên Transformer kết hợp Phân tích Cảm xúc Tin tức

Báo cáo Niên luận — Học phần **Thực tập viết niên luận (TIN3142)**
Khoa Công nghệ Thông tin — Trường Đại học Khoa học, Đại học Huế

- **Sinh viên thực hiện:** Trần Hữu Minh Duy — MSV: 23T1020134 — Lớp CNTT K47I
- **Giảng viên hướng dẫn:** Lê Quang Chiến

Repository này chứa toàn bộ mã nguồn để tái tạo (reproduce) các số liệu và biểu đồ
trong báo cáo: thu thập dữ liệu OHLCV (Binance) + tiêu đề tin tức (CoinDesk,
Cointelegraph, The Block), tiền xử lý, huấn luyện mô hình **Sentiment-Augmented
Transformer (Two-Modal)**, và đánh giá trên tập kiểm thử.

---

## 1. Kiến trúc tổng quan

```
OHLCV (48h) ──► Transformer Encoder (6 layers, 12 heads) ──► H_tech ─┐
                                                                       ├──► Gated Fusion ──► Classifier ──► Tăng/Giảm (24h)
Sentiment (48h) ──► Linear Projection (1 → 768) ──────────► H_sent ──┘
```

Chi tiết công thức và sơ đồ đầy đủ xem tại Chương 3 của báo cáo (`2026.2.N16_..._Tuan5.pdf`).

## 2. Cấu trúc thư mục

```
.
├── configs/
│   └── config.yaml              # Toàn bộ hyperparameter (khớp Bảng 3.2 / 4.3 báo cáo)
├── data/
│   ├── raw/                     # Dữ liệu thô sau khi tải (không commit lên git)
│   └── processed/               # Dataset đặc trưng cuối cùng (17 features + target)
├── src/
│   ├── data_collection/
│   │   ├── fetch_binance_ohlcv.py   # Thu thập OHLCV từ Binance API
│   │   └── fetch_news.py            # Crawl tiêu đề tin tức CoinDesk/Cointelegraph/The Block
│   ├── preprocessing/
│   │   ├── technical_indicators.py  # RSI, MACD, Bollinger Bands, EMA, Volatility
│   │   ├── sentiment_analysis.py    # DistilBERT sentiment scoring
│   │   └── build_dataset.py         # Ghép + làm sạch + gán nhãn (Algorithm 1)
│   ├── models/
│   │   └── two_modal_model.py       # Module 1-4: Encoder, Projection, Gated Fusion, Classifier
│   ├── dataset.py                # Chronological split + Min-Max scaler + sliding window Dataset
│   ├── train.py                  # Vòng lặp huấn luyện (AdamW, FP16, early stopping)
│   ├── evaluate.py               # Đánh giá Test set: metrics + Confusion Matrix + ROC
│   └── utils.py
├── checkpoints/                  # Model checkpoint tốt nhất (.pt) — không commit
├── figures/                      # Biểu đồ sinh ra sau train/evaluate — không commit
├── requirements.txt
└── README.md
```

## 3. Cài đặt môi trường

Yêu cầu: **Python 3.9+**. Khuyến nghị dùng GPU có CUDA (báo cáo dùng GPU 4GB VRAM,
xem mục 4.1.1), nhưng toàn bộ code vẫn chạy được trên CPU (chậm hơn).

```bash
# 1. Clone repository
git clone https://github.com/<username>/thuc-tap-nien-luan.git
cd thuc-tap-nien-luan

# 2. Tạo virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

> Nếu dùng GPU, cài thêm đúng bản PyTorch có hỗ trợ CUDA phù hợp với driver máy bạn
> theo hướng dẫn tại https://pytorch.org/get-started/locally/ (thay vì bản CPU mặc định
> trong `requirements.txt`).

## 4. Chuẩn bị dữ liệu (Dataset)

### Cách 1 — Thu thập lại từ đầu (đúng quy trình trong báo cáo)

```bash
# Bước 1: Tải dữ liệu OHLCV BTC/USDT (hourly) từ Binance
python -m src.data_collection.fetch_binance_ohlcv \
    --symbol BTCUSDT --interval 1h \
    --start 2024-01-01 --end 2026-07-01 \
    --out data/raw/ohlcv_raw.csv

# Bước 2: Crawl tiêu đề tin tức Crypto
python -m src.data_collection.fetch_news \
    --out data/raw/news_raw.csv --pages 50

# Bước 3: Tiền xử lý, tính chỉ báo kỹ thuật, sentiment, ghép & gán nhãn
python -m src.preprocessing.build_dataset \
    --ohlcv data/raw/ohlcv_raw.csv \
    --news data/raw/news_raw.csv \
    --out data/processed/dataset_features.csv
```

> **Lưu ý:** `fetch_news.py` dùng selector HTML đơn giản để minh họa quy trình crawl;
> cấu trúc trang tin thay đổi theo thời gian nên có thể cần chỉnh lại selector, hoặc
> thay bằng API tổng hợp tin tức (vd. CryptoPanic, NewsAPI) để lấy dữ liệu lịch sử ổn định hơn.

### Cách 2 — Dùng dữ liệu mẫu để test nhanh pipeline

Repo kèm sẵn 2 file **dữ liệu tổng hợp (synthetic)** — KHÔNG phải dữ liệu thị trường
thật — chỉ nhằm mục đích kiểm tra pipeline chạy đúng end-to-end trước khi thu thập
dữ liệu thật đầy đủ:

```
data/raw/ohlcv_sample.csv      # 2,000 nến giả lập (random walk)
data/raw/news_sample.csv       # 400 tiêu đề mẫu (positive/negative/neutral)
```

```bash
python -m src.preprocessing.build_dataset \
    --ohlcv data/raw/ohlcv_sample.csv \
    --news data/raw/news_sample.csv \
    --out data/processed/dataset_features.csv
```

Sau khi xác nhận pipeline chạy thông (không lỗi), thay bằng dữ liệu thật lấy từ
Cách 1 để huấn luyện mô hình cho kết quả có ý nghĩa thực tế.

Sau khi hoàn tất, `data/processed/dataset_features.csv` sẽ có đúng cấu trúc
17 trường theo Bảng 2.2 của báo cáo: 5 OHLCV + 11 chỉ báo kỹ thuật + 1 sentiment
theo giờ + 1 nhãn Target.

## 5. Huấn luyện mô hình (Training)

```bash
python -m src.train --config configs/config.yaml
```

Quá trình huấn luyện sẽ:
- Chia dữ liệu theo **Chronological Split 70/15/15** (không xáo trộn, chống Data Leakage — mục 2.2.4)
- Fit `MinMaxScaler` **chỉ trên tập Train**
- Huấn luyện tối đa 50 epochs với **AdamW** (lr=1e-4, weight_decay=1e-4), **Mixed
  Precision (FP16)**, **Gradient Accumulation** (2 bước, mô phỏng batch 64), **Gradient
  Clipping** (norm=1.0), và **Early Stopping** (patience=10 epochs) — khớp mục 3.3.4 / Bảng 4.3
- Lưu checkpoint tốt nhất vào `checkpoints/best_model.pt`
- Xuất biểu đồ hội tụ Loss/Accuracy vào `figures/loss_curve.png`, `figures/accuracy_curve.png`
  (tương ứng Hình 4.1, 4.2 của báo cáo)

Có thể chỉnh mọi hyperparameter (lookback window, số layer, learning rate, batch size...)
trực tiếp trong `configs/config.yaml` mà không cần sửa code.

## 6. Đánh giá mô hình (Evaluation)

```bash
python -m src.evaluate --config configs/config.yaml
```

Script này nạp checkpoint tốt nhất, dự báo trên tập Test độc lập, và xuất ra:
- Accuracy, Precision, Recall, F1-Score, AUC-ROC, Specificity (in ra console + lưu `figures/test_metrics.json`)
- Confusion Matrix dạng heatmap → `figures/confusion_matrix.png` (Hình 4.3)
- Đường cong ROC → `figures/roc_curve.png` (Hình 4.4)

Các số liệu này tương ứng trực tiếp với Bảng 4.4, 4.5 và Hình 4.3, 4.4 trong báo cáo.

## 7. Tái tạo toàn bộ kết quả từ đầu đến cuối

```bash
pip install -r requirements.txt

python -m src.data_collection.fetch_binance_ohlcv --start 2024-01-01 --end 2026-07-01 --out data/raw/ohlcv_raw.csv
python -m src.data_collection.fetch_news --out data/raw/news_raw.csv
python -m src.preprocessing.build_dataset --ohlcv data/raw/ohlcv_raw.csv --news data/raw/news_raw.csv --out data/processed/dataset_features.csv

python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
```

Toàn bộ số liệu, biểu đồ sẽ xuất hiện trong `figures/` sau khi chạy xong hai lệnh cuối.

> **Ghi chú về khả năng tái lập:** vì dữ liệu thị trường và tin tức thay đổi liên tục,
> kết quả số cụ thể (Accuracy, F1...) khi crawl lại dữ liệu mới có thể chênh lệch vài
> điểm phần trăm so với số liệu đã báo cáo (Accuracy 70.23%, F1 70.50%, AUC-ROC 0.7634),
> do khoảng thời gian dữ liệu và các sự kiện thị trường khác nhau. Để tái tạo **chính xác**
> số liệu trong báo cáo, cần dùng đúng khoảng dữ liệu 01/01/2024–01/07/2026 đã nêu ở mục 1.2.2.

## 8. Giấy phép & Trích dẫn

Mã nguồn phục vụ mục đích học thuật (Niên luận TIN3142). Nếu tham khảo, vui lòng trích
dẫn báo cáo đi kèm.
