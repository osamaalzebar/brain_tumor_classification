

import os
import csv
import json
import argparse
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset_vgg16 import BrainMRIDataset

DEFAULT_CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]


def build_model(num_classes: int = 4) -> torch.nn.Module:
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    return ckpt


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_targets = []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        all_logits.append(outputs.detach().cpu())
        all_targets.append(targets.detach().cpu())
    logits = torch.cat(all_logits, dim=0).numpy() if all_logits else np.zeros((0, 4), dtype=np.float32)
    targets = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0,), dtype=np.int64)
    preds = logits.argmax(axis=1) if len(logits) else np.array([], dtype=np.int64)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy() if len(logits) else np.zeros((0, 4), dtype=np.float32)
    max_probs = probs.max(axis=1) if len(probs) else np.array([], dtype=np.float32)
    return preds, targets, max_probs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4):
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(cm[c, :].sum())
        per_class.append((prec, rec, f1, support))
    return acc, cm, per_class


def save_confusion_matrix_png(cm, class_names: List[str], out_png: str):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_predictions_csv(csv_in: str, out_csv: str, preds0: np.ndarray, probs_max: np.ndarray):
    rows = []
    with open(csv_in, "r", newline="") as f:
        reader = csv.DictReader(f)
        assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames, "CSV must have Image_path,label"
        for row in reader:
            rows.append(row)
    if len(rows) != len(preds0):
        raise RuntimeError(f"CSV rows ({len(rows)}) != predictions ({len(preds0)})")

    fieldnames = ["Image_path", "true_label_1to4", "pred_label_1to4", "pred_confidence"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r, p0, conf in zip(rows, preds0, probs_max):
            writer.writerow({
                "Image_path": r["Image_path"],
                "true_label_1to4": int(r["label"]),
                "pred_label_1to4": int(p0 + 1),
                "pred_confidence": float(conf),
            })


def parse_args():
    p = argparse.ArgumentParser(description="Test fine-tuned VGG16 on MRI brain classification (4 classes).")
    p.add_argument("--root", type=str, required=True, help="Directory containing test images.")
    p.add_argument("--csv", type=str, required=True, help="CSV with Image_path,label (labels in 1..4).")
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pth).")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="test_outputs")
    p.add_argument("--class-names", type=str, nargs="*", default=DEFAULT_CLASS_NAMES,
                   help="Optional override for class names in order 1..4.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = BrainMRIDataset(args.root, args.csv, img_size=args.image_size, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=4).to(device)
    _ = load_checkpoint(model, args.ckpt, map_location=device)

    preds, targets, max_probs = run_inference(model, test_loader, device)
    acc, cm, per_class = compute_metrics(targets, preds, num_classes=4)

    print(f"Overall accuracy: {acc:.4f}")
    for i, (prec, rec, f1, support) in enumerate(per_class, start=1):
        name = args.class_names[i-1] if i-1 < len(args.class_names) else f"class_{i}"
        print(f"[{i}] {name:>10s} | precision: {prec:.4f} | recall: {rec:.4f} | f1: {f1:.4f} | support: {support}")

    preds_csv = os.path.join(args.out_dir, "predictions.csv")
    metrics_json = os.path.join(args.out_dir, "metrics.json")
    cm_png = os.path.join(args.out_dir, "confusion_matrix.png")

    write_predictions_csv(args.csv, preds_csv, preds, max_probs)
    save_confusion_matrix_png(cm, args.class_names, cm_png)

    with open(metrics_json, "w") as f:
        json.dump({
            "overall_accuracy": float(acc),
            "per_class": [
                {
                    "class_index_1to4": i + 1,
                    "class_name": args.class_names[i] if i < len(args.class_names) else f"class_{i+1}",
                    "precision": float(per_class[i][0]),
                    "recall": float(per_class[i][1]),
                    "f1": float(per_class[i][2]),
                    "support": int(per_class[i][3]),
                } for i in range(4)
            ],
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    print(f"Saved predictions to: {preds_csv}")
    print(f"Saved confusion matrix image to: {cm_png}")
    print(f"Saved metrics JSON to: {metrics_json}")


if __name__ == "__main__":
    main()
