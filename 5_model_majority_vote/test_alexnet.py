# test_alexnet_brain_tumor.py
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights

from dataset_alexnet import MRIBrainTumorCSV, build_transforms, CLASS_NAMES


def build_alexnet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Rebuild AlexNet with last 3 FC layers set for `num_classes`, matching training."""
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)

    in_features = model.classifier[1].in_features  # typically 9216
    hidden = 4096
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes),
    )
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[List[float]]]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_probs: List[List[float]] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_true.extend(targets.tolist())
        all_pred.extend(preds.tolist())
        all_probs.extend(probs.cpu().tolist())

    return all_true, all_pred, all_probs


def confusion_matrix(true: List[int], pred: List[int], k: int = 4):
    cm = [[0 for _ in range(k)] for _ in range(k)]
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def per_class_prf(cm: List[List[int]]):
    k = len(cm)
    out = []
    for i in range(k):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k)) - tp
        fn = sum(cm[i][c] for c in range(k)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append((prec, rec, f1))
    return out


def read_csv_image_paths(csv_path: str) -> List[str]:
    paths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            paths.append(row[0].strip())
    return paths


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned AlexNet on 4-class brain tumor dataset")
    parser.add_argument("--test_dir", type=str,
                        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/data",
                        help="Root folder for TEST images (filenames referenced in CSV)")
    parser.add_argument("--test_csv", type=str,
                        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/image_labels_test.csv",
                        help="TEST CSV path with header: Image_path,label (labels 1..4)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/alexnet_best.pth",
                        help="Checkpoint saved by the training script (.pth/.pt)")
    parser.add_argument("--img_size", type=int, default=224)  # use 227 if you trained that way
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet init (state_dict still loads)")
    parser.add_argument("--save_preds", type=str, default="preds_alexnet_test.csv",
                        help="Output CSV file for per-image predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / loader — uses same normalization & grayscale->RGB as training
    test_tfms = build_transforms(args.img_size, train=False)
    test_ds = MRIBrainTumorCSV(args.test_dir, args.test_csv, transform=test_tfms)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model (same head as training)
    model = build_alexnet(num_classes=4, pretrained=not args.no_pretrained).to(device)

    # Load checkpoint (handles raw state_dict OR a dict with "model_state")
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing: {missing}")
        print(f"[load_state_dict] unexpected: {unexpected}")

    # Inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # Metrics
    total = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    acc = correct / total if total > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred, k=4)
    prf = per_class_prf(cm)

    print("\n=== Test Results (AlexNet) ===")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    for row in cm:
        print("  " + " ".join(f"{v:5d}" for v in row))

    print("\nPer-class (precision, recall, f1):")
    for i, (prec, rec, f1) in enumerate(prf):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i} ({name}): P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Save per-image predictions
    try:
        img_paths = read_csv_image_paths(args.test_csv)
        n = min(len(img_paths), total)
        out_p = Path(args.save_preds)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "true_idx(0-3)", "pred_idx(0-3)", "true_label(1-4)", "pred_label(1-4)"] \
                     + [f"prob_{i}" for i in range(4)]
            writer.writerow(header)
            for i in range(n):
                writer.writerow([
                    img_paths[i],
                    y_true[i],
                    y_pred[i],
                    y_true[i] + 1,  # back to 1..4 for convenience
                    y_pred[i] + 1,
                    *[f"{p:.6f}" for p in y_prob[i]]
                ])
        print(f"\nSaved predictions to: {out_p.resolve()}")
    except Exception as e:
        print(f"[warn] Could not save predictions CSV: {e}")


if __name__ == "__main__":
    main()
