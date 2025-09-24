#!/usr/bin/env python3
# test_inceptionv3_concat.py

import argparse
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from train_inceptionV3 import InceptionV3ConcatHead   # import your model
from dataset_inceptionV3 import BrainCSVSet                    # dataset + transforms

# -----------------------------
# Utilities
# -----------------------------
def accuracy_top1(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

# -----------------------------
# Main
# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Test fine-tuned InceptionV3Concat on test set")
    ap.add_argument("--test-root", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/data")
    ap.add_argument("--test-csv",  type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/image_labels.csv")
    ap.add_argument("--checkpoint", type=str, default="./checkpoint/best_incv3_concat.pth")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", type=str, default="test_predictions_incv3.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Dataset / Loader
    test_ds = BrainCSVSet(args.test_root, args.test_csv, train=False, img_size=299)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = InceptionV3ConcatHead(num_classes=4)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # Test loop
    total, correct = 0, 0
    rows = []

    for imgs, labels, paths in test_dl:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        preds = logits.argmax(dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        probs = F.softmax(logits, dim=1)

        # collect rows
        for i in range(labels.size(0)):
            rows.append({
                "image": paths[i],
                "true_label": int(labels[i].item()),
                "pred_label": int(preds[i].item()),
                "prob_c0": float(probs[i,0].item()),
                "prob_c1": float(probs[i,1].item()),
                "prob_c2": float(probs[i,2].item()),
                "prob_c3": float(probs[i,3].item()),
            })

    acc = correct / total if total else 0.0

    # Save CSV
    out = Path(args.out_csv)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["image","true_label","pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to {out.resolve()}")
    print(f"Total samples: {total}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
