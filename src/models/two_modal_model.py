"""
Kiến trúc Sentiment-Augmented Transformer (Two-Modal Model).
Khớp mục 3.2 của báo cáo:
  Module 1: Transformer Encoder trích xuất đặc trưng kỹ thuật
  Module 2: Tuyến tính hóa đặc trưng cảm xúc (Sentiment Projection)
  Module 3: Gated Fusion Layer
  Module 4: Bộ phân loại xu hướng giá (Classifier)
"""
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """Positional Encoding học được (thay vì sin/cos cố định) — mục 3.2.2."""

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pos_embedding[:, : x.size(1), :]


class TechnicalEncoder(nn.Module):
    """Module 1: Linear Embedding + Learnable PE + Transformer Encoder Stack."""

    def __init__(self, n_features, d_model, n_layers, n_heads, ffn_dim, dropout, seq_len):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = LearnablePositionalEncoding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x_tech):  # x_tech: (B, T, n_features)
        x = self.embedding(x_tech)          # (B, T, d_model)
        x = self.pos_encoding(x)
        h_tech = self.encoder(x)            # (B, T, d_model)
        return h_tech


class SentimentProjection(nn.Module):
    """Module 2: chiếu tuyến tính điểm sentiment (1 chiều) lên d_model chiều, mỗi timestep."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, x_sent):  # x_sent: (B, T, 1)
        return self.proj(x_sent)  # (B, T, d_model)


class GatedFusionLayer(nn.Module):
    """
    Module 3: cổng Sigmoid học tại mỗi timestep để trộn động Htech và H'sent,
    sau đó Global Average Pooling theo chiều thời gian. Khớp công thức mục 3.2.4.
    """

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 1)

    def forward(self, h_tech, h_sent):  # cả hai: (B, T, d_model)
        concat = torch.cat([h_tech, h_sent], dim=-1)      # (B, T, 2*d_model)
        g = torch.sigmoid(self.gate(concat))                # (B, T, 1)
        h_context = g * h_tech + (1 - g) * h_sent            # (B, T, d_model)
        h_fused = h_context.mean(dim=1)                      # Global Average Pooling -> (B, d_model)
        return h_fused, g.squeeze(-1)  # trả thêm gate để phục vụ visualization sau này


class TrendClassifier(nn.Module):
    """Module 4: MLP 3 lớp -> Softmax nhị phân (Tăng/Giảm)."""

    def __init__(self, d_model, hidden_dims, dropout_rates, n_classes=2):
        super().__init__()
        h1, h2 = hidden_dims
        d1, d2 = dropout_rates
        self.net = nn.Sequential(
            nn.Linear(d_model, h1),
            nn.ReLU(),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(d2),
            nn.Linear(h2, n_classes),
        )

    def forward(self, h_fused):
        return self.net(h_fused)  # logits (B, n_classes)


class TwoModalTransformer(nn.Module):
    """Ghép toàn bộ 4 module thành mô hình end-to-end."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        seq_len = cfg["data"]["lookback_window"]

        self.technical_encoder = TechnicalEncoder(
            n_features=m["n_technical_features"],
            d_model=m["d_model"],
            n_layers=m["n_encoder_layers"],
            n_heads=m["n_attention_heads"],
            ffn_dim=m["ffn_hidden_dim"],
            dropout=m["encoder_dropout"],
            seq_len=seq_len,
        )
        self.sentiment_projection = SentimentProjection(m["d_model"])
        self.fusion = GatedFusionLayer(m["d_model"])
        self.classifier = TrendClassifier(
            d_model=m["d_model"],
            hidden_dims=m["classifier_hidden"],
            dropout_rates=m["classifier_dropout"],
            n_classes=m["n_classes"],
        )

    def forward(self, x_tech, x_sent):
        h_tech = self.technical_encoder(x_tech)          # (B, T, d_model)
        h_sent = self.sentiment_projection(x_sent)         # (B, T, d_model)
        h_fused, gate_weights = self.fusion(h_tech, h_sent)  # (B, d_model), (B, T)
        logits = self.classifier(h_fused)                   # (B, 2)
        return logits, gate_weights

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
