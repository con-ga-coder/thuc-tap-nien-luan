"""
Huấn luyện mô hình Two-Modal Transformer.
Khớp mục 3.3.4 và Bảng 4.3 của báo cáo:
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
  - Mixed Precision Training (FP16), Gradient Accumulation (2 steps)
  - Gradient Clipping = 1.0
  - Early Stopping (patience = 10 epochs), max 50 epochs

Chạy:
    python src/train.py --config configs/config.yaml
"""
import argparse
import copy
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_splits
from src.models.two_modal_model import TwoModalTransformer
from src.utils import load_config, set_seed, get_device


def run_epoch(model, loader, criterion, optimizer, device, scaler, accum_steps, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    if train:
        optimizer.zero_grad()

    for step, (x_tech, x_sent, y) in enumerate(loader):
        x_tech, x_sent, y = x_tech.to(device), x_sent.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(scaler is not None)):
            logits, _ = model(x_tech, x_sent)
            loss = criterion(logits, y) / (accum_steps if train else 1)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        batch_loss = loss.item() * (accum_steps if train else 1)
        total_loss += batch_loss * y.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (Cross-Entropy)")
    plt.title("Đường cong hội tụ: Training vs Validation Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{out_dir}/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Đường cong hội tụ: Training vs Validation Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(f"{out_dir}/accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    device = get_device()
    print(f"Sử dụng thiết bị: {device}")

    train_ds, val_ds, _, _ = build_splits(
        cfg["data"]["processed_path"],
        lookback=cfg["data"]["lookback_window"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

    model = TwoModalTransformer(cfg).to(device)
    print(f"Tổng số tham số khả vi: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["mixed_precision"] and device.type == "cuda")

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_scaler, cfg["training"]["gradient_accumulation_steps"], train=True,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            grad_scaler, 1, train=False,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_state, cfg["training"]["checkpoint_path"])
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                print(f"Early Stopping tại epoch {epoch} (không cải thiện sau "
                      f"{cfg['training']['early_stopping_patience']} epochs).")
                break

    plot_curves(history, cfg["evaluation"]["figures_dir"])
    with open("checkpoints/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Huấn luyện hoàn tất. Checkpoint tốt nhất: {cfg['training']['checkpoint_path']}")


if __name__ == "__main__":
    main()
