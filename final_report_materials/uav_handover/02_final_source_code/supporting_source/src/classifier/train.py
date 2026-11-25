"""
Patch classifier training loop.

Usage:
    python -m src.classifier.train --config configs/config.yaml

What it does:
    1. Loads PatchDataset for train/val/test splits.
    2. Instantiates MultiscalePatchClassifier (DINOv2 ViT-S/14, 9-channel).
    3. Trains with:
        - Backbone frozen for first `warmup_epochs` epochs.
        - AdamW with separate LRs for backbone and head.
        - Cosine LR schedule.
        - SoftCE + PairwiseMargin loss.
        - Class-balanced WeightedRandomSampler.
    4. Validates each epoch: macro-F1, per-class F1, per-confusion-pair error.
    5. Saves best checkpoint to runs/classifier/best.pt.
    6. Writes training_report.json + confusion matrix PNG on completion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.classifier.dataset import PatchDataset
from src.classifier.losses import ClassifierLoss
from src.classifier.model import MultiscalePatchClassifier
from src.utils.constants import CLASS_NAMES, CONFUSION_PAIRS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    all_preds: list[int],
    all_labels: list[int],
) -> dict:
    """
    Compute:
        - Per-class precision, recall, F1.
        - Macro-F1.
        - Per-confusion-pair error rate.
        - Overall accuracy.
    """
    from collections import defaultdict
    n = NUM_CLASSES
    tp = np.zeros(n)
    fp = np.zeros(n)
    fn = np.zeros(n)
    correct = 0

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1        = np.where(precision + recall > 0,
                         2 * precision * recall / (precision + recall), 0.0)
    macro_f1  = f1.mean()
    accuracy  = correct / max(len(all_labels), 1)

    from src.utils.constants import CLASS_TO_IDX
    confusion_pair_errors = {}
    for cls_a, cls_b in CONFUSION_PAIRS:
        idx_a = CLASS_TO_IDX.get(cls_a, -1)
        idx_b = CLASS_TO_IDX.get(cls_b, -1)
        if idx_a < 0 or idx_b < 0:
            continue
        # Error rate: fraction of class_a samples predicted as class_b + vice versa
        total_ab = sum(1 for l in all_labels if l in (idx_a, idx_b))
        confused  = sum(
            1 for p, l in zip(all_preds, all_labels)
            if (l == idx_a and p == idx_b) or (l == idx_b and p == idx_a)
        )
        err = confused / max(total_ab, 1)
        confusion_pair_errors[f"{cls_a}↔{cls_b}"] = round(err, 4)

    per_class = {
        CLASS_NAMES[i]: {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1[i]), 4),
        }
        for i in range(n)
    }

    return {
        "accuracy":             round(accuracy, 4),
        "macro_f1":             round(float(macro_f1), 4),
        "per_class":            per_class,
        "confusion_pair_errors": confusion_pair_errors,
    }


def save_confusion_matrix(
    all_preds: list[int],
    all_labels: list[int],
    out_path: Path,
) -> None:
    """Save a confusion matrix PNG using matplotlib."""
    import matplotlib.pyplot as plt

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l, p] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)
    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = config["classifier"]
    paths = config["paths"]
    out_dir = Path(paths["runs_dir"]) / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir / "tb_logs")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    patches_dir = paths["patches_dir"]

    ds_train = PatchDataset(
        patches_dir=patches_dir,
        split="train",
        val_split=cfg["val_split"],
        test_split=cfg["test_split"],
        seed=cfg["seed"],
        input_size=cfg["input_size"],
        mixup_prob=0.25,
        cutmix_prob=0.25,
        num_classes=NUM_CLASSES,
    )
    ds_val = PatchDataset(
        patches_dir=patches_dir,
        split="val",
        val_split=cfg["val_split"],
        test_split=cfg["test_split"],
        seed=cfg["seed"],
        input_size=cfg["input_size"],
        num_classes=NUM_CLASSES,
    )
    ds_test = PatchDataset(
        patches_dir=patches_dir,
        split="test",
        val_split=cfg["val_split"],
        test_split=cfg["test_split"],
        seed=cfg["seed"],
        input_size=cfg["input_size"],
        num_classes=NUM_CLASSES,
    )

    print(f"  train: {len(ds_train)}  val: {len(ds_val)}  test: {len(ds_test)}")
    class_counts = ds_train.get_class_counts()
    print("  class counts (train):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:15} {class_counts.get(i, 0):>4}")

    sampler = ds_train.get_weighted_sampler(
        oversample_barren=cfg.get("oversample_barren", 8)
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nBuilding model...")
    weights_path = cfg.get("pretrained_weights_path")
    model = MultiscalePatchClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        pretrained_weights_path=weights_path,
        img_size=cfg["input_size"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Total: {total_params:.1f}M  Trainable: {trainable_params:.1f}M")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = ClassifierLoss(
        class_counts=class_counts,
        barren_oversample=cfg.get("oversample_barren", 8),
        margin_weight=cfg.get("margin_loss_weight", 0.3),
        margin=cfg.get("margin_loss_alpha", 0.3),
    ).to(device)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    warmup_epochs = cfg.get("warmup_epochs", 5)
    model.freeze_backbone()
    print(f"  Backbone frozen for first {warmup_epochs} epochs.")

    optimizer = torch.optim.AdamW(
        model.get_optimizer_param_groups(
            lr_head=cfg["lr_head"],
            lr_backbone=cfg["lr_backbone"],
        ),
        weight_decay=cfg["weight_decay"],
    )

    total_epochs = cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_f1 = 0.0
    best_ckpt_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, total_epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == warmup_epochs + 1:
            model.unfreeze_backbone()
            print(f"\nEpoch {epoch}: backbone unfrozen.")
            # Reset optimizer to apply backbone LR properly
            optimizer = torch.optim.AdamW(
                model.get_optimizer_param_groups(
                    lr_head=cfg["lr_head"],
                    lr_backbone=cfg["lr_backbone"],
                ),
                weight_decay=cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - epoch,
                eta_min=1e-6,
            )

        # ---- Train ----
        model.train()
        train_losses = {"loss": 0.0, "ce": 0.0, "margin": 0.0}
        t0 = time.time()

        for batch in tqdm(dl_train, desc=f"Epoch {epoch:3d}/{total_epochs} [train]",
                          leave=False):
            images      = batch["image"].to(device)       # (B, 9, H, W)
            soft_labels = batch["soft_label"].to(device)  # (B, C)
            hard_labels = batch["label"].to(device)       # (B,)
            orig_sizes  = batch["orig_size"]               # (B,) int — stays CPU

            optimizer.zero_grad()
            logits = model(images, orig_sizes)
            embeddings = model.get_embeddings(images, orig_sizes)
            loss_dict = criterion(logits, embeddings, soft_labels, hard_labels)
            loss_dict["loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += loss_dict[k].item()

        if epoch > warmup_epochs:
            scheduler.step()

        n_batches = len(dl_train)
        for k in train_losses:
            train_losses[k] /= n_batches

        # ---- Validate ----
        model.eval()
        all_preds, all_labels_list = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Epoch {epoch:3d}/{total_epochs} [val]  ",
                              leave=False):
                images      = batch["image"].to(device)
                soft_labels = batch["soft_label"].to(device)
                hard_labels = batch["label"].to(device)
                orig_sizes  = batch["orig_size"]

                logits = model(images, orig_sizes)
                embeddings = model.get_embeddings(images, orig_sizes)
                ld = criterion(logits, embeddings, soft_labels, hard_labels)
                val_loss += ld["loss"].item()

                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels_list.extend(hard_labels.cpu().tolist())

        val_loss /= max(len(dl_val), 1)
        metrics = compute_metrics(all_preds, all_labels_list)
        val_f1 = metrics["macro_f1"]
        elapsed = time.time() - t0

        # ---- Logging ----
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"train_loss={train_losses['loss']:.4f} "
            f"(ce={train_losses['ce']:.4f} margin={train_losses['margin']:.4f}) | "
            f"val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f} | "
            f"acc={metrics['accuracy']:.4f} | {elapsed:.0f}s"
        )
        print("  confusion pairs:", metrics["confusion_pair_errors"])

        writer.add_scalar("Loss/train", train_losses["loss"], epoch)
        writer.add_scalar("Loss/val",   val_loss, epoch)
        writer.add_scalar("F1/val_macro", val_f1, epoch)
        for name, m in metrics["per_class"].items():
            writer.add_scalar(f"F1_per_class/val_{name}", m["f1"], epoch)

        # ---- Checkpoint ----
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_losses["loss"],
            "val_loss": val_loss,
            **metrics,
        }
        history.append(epoch_record)

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "metrics": metrics,
                    "config": config,
                },
                best_ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint (val_macro_f1={val_f1:.4f})")

    # ------------------------------------------------------------------
    # Final test evaluation on best checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Test eval"):
            images     = batch["image"].to(device)
            hard_labels = batch["label"].to(device)
            orig_sizes  = batch["orig_size"]
            logits = model(images, orig_sizes)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels_list.extend(hard_labels.cpu().tolist())

    test_metrics = compute_metrics(all_preds, all_labels_list)
    print(f"\nTest Results:")
    print(f"  macro_f1 = {test_metrics['macro_f1']:.4f}")
    print(f"  accuracy = {test_metrics['accuracy']:.4f}")
    print("  per-class F1:")
    for cls, m in test_metrics["per_class"].items():
        print(f"    {cls:15} {m['f1']:.4f}")
    print("  confusion pair errors:")
    for pair, err in test_metrics["confusion_pair_errors"].items():
        flag = "⚠" if err > 0.07 else "✓"
        print(f"    {flag} {pair}: {err:.4f}")

    # Save confusion matrix PNG
    cm_path = out_dir / "confusion_matrix_test.png"
    save_confusion_matrix(all_preds, all_labels_list, cm_path)
    print(f"  Saved: {cm_path}")

    # Save full report
    report = {
        "best_val_macro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
        "config": config,
    }
    report_path = out_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    writer.close()
    print(f"\nDone. Best val macro-F1: {best_val_f1:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
