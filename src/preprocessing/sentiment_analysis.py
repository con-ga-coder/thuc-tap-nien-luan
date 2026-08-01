"""
Phân tích cảm xúc tiêu đề tin tức bằng mô hình pre-trained
distilbert-base-uncased-finetuned-sst-2-english.
Tương ứng mục 2.2.2 (Xử lý luồng văn bản tin tức) của báo cáo.

Sentiment Score = P(positive) - P(negative)  ->  khoảng [-1, 1]
"""
import re

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # loại bỏ thẻ HTML
    text = re.sub(r"http\S+", " ", text)            # loại bỏ URL
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)      # loại bỏ ký tự đặc biệt/emoji
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class SentimentScorer:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # id2label thường là {0: 'NEGATIVE', 1: 'POSITIVE'} cho model sst-2
        self.id2label = {int(k): v.upper() for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def score_batch(self, texts: list, batch_size: int = 32) -> list:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                p_pos = sum(row[i] for i, lab in self.id2label.items() if lab == "POSITIVE")
                p_neg = sum(row[i] for i, lab in self.id2label.items() if lab == "NEGATIVE")
                scores.append(float(p_pos - p_neg))
        return scores


def score_news_dataframe(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    df = df.copy()
    df["title_clean"] = df[title_col].astype(str).apply(clean_text)
    scorer = SentimentScorer()
    df["sentiment_score"] = scorer.score_batch(df["title_clean"].tolist())
    return df
