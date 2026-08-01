"""
Đánh giá mô hình trên tập Test độc lập.
Sinh ra các bảng/biểu đồ tương ứng mục 4.3 của báo cáo:
  - Accuracy, Precision, Recall, F1-Score, AUC-ROC, Specificity
  - Confusion Matrix + Heatmap
  - Đường cong ROC

Chạy:
    python src/evaluate.py --config configs/config.yaml
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from torch.utils.data import DataLoader

from src.dataset import build_splits
from src.models.two_modal_model import TwoModalTransformer
from src.utils import load_config, get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for x_tech, x_sent, y in loader:
        x_tech, x_sent = x_tech.to(device), x_sent.to(device)
        logits, _ = model(x_tech, x_sent)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(tăng giá)
        preds = logits.argmax(dim=-1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Giảm (0)", "Tăng (1)"],
                yticklabels=["Giảm (0)", "Tăng (1)"])
    plt.xlabel("Dự báo"); plt.ylabel("Thực tế")
    plt.title("Ma trận Nhầm lẫn (Confusion Matrix) - Tập Test")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_prob, auc, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Two-Modal (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="red", label="Random Classifier (AUC = 0.5)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Đường cong ROC (Receiver Operating Characteristic)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    _, _, test_ds, _ = build_splits(
        cfg["data"]["processed_path"],
        lookback=cfg["data"]["lookback_window"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = TwoModalTransformer(cfg).to(device)
    model.load_state_dict(torch.load(cfg["training"]["checkpoint_path"], map_location=device))

    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp)

    print("=== Kết quả trên Test Set ===")
    for k, v in metrics.items():
        print(f"{k:12s}: {v*100:.2f}%")
    print("Confusion Matrix:\n", cm)

    fig_dir = cfg["evaluation"]["figures_dir"]
    plot_confusion_matrix(cm, f"{fig_dir}/confusion_matrix.png")
    plot_roc(y_true, y_prob, metrics["auc_roc"], f"{fig_dir}/roc_curve.png")

    with open(f"{fig_dir}/test_metrics.json", "w") as f:
        json.dump({**metrics, "confusion_matrix": cm.tolist()}, f, indent=2)

    print(f"Đã lưu biểu đồ và metrics vào {fig_dir}")


if __name__ == "__main__":
    main()
