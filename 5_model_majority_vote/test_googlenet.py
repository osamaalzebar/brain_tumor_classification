# test_googlenet_brain_tumor.py
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights

from dataset_densenet import MRIDataset, build_transforms  # same as training


# --------- Model builder (identical to training) ----------
def build_googlenet(num_classes: int = 4, pretrained: bool = True, aux_logits: bool = True):
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights, aux_logits=aux_logits)

    # Replace primary classifier (loss3-classifier)
    model.fc = nn.Linear(1024, num_classes)

    # Replace auxiliary classifiers
    if aux_logits:
        model.aux1.fc = nn.Linear(1024, num_classes)
        model.aux2.fc = nn.Linear(1024, num_classes)

    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4
) -> Tuple[List[int], List[int], List[List[float]]]:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(targets.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    return all_targets, all_preds, all_probs


def confusion_matrix(true: List[int], pred: List[int], num_classes: int = 4):
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def per_class_prf(cm: List[List[int]]):
    k = len(cm)
    metrics = []
    for i in range(k):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(k)) - tp
        fn = sum(cm[i][c] for c in range(k)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics.append((prec, rec, f1))
    return metrics


def read_csv_image_paths(csv_path: str) -> List[str]:
    paths = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Expecting something like: Image_path,label
        # We'll use the first column as the image path.
        for row in reader:
            if not row:
                continue
            paths.append(row[0])
    return paths


def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned GoogLeNet on 4-class brain tumor MRI")
    # Defaults set to your provided test locations
    parser.add_argument("--test_image_dir", type=str,
                        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/data",
                        help="Root folder for TEST images")
    parser.add_argument("--test_csv", type=str,
                        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/image_labels_test.csv",
                        help="TEST CSV (Image_path,label)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/googlenet_best.pth",
                        help="Path to trained checkpoint (.pth) saved by training script")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet init (doesn't matter when loading checkpoint weights)")
    parser.add_argument("--no_aux_logits", action="store_true",
                        help="Disable aux logits for faster inference (checkpoint still loads fine with strict=False)")
    parser.add_argument("--save_preds", type=str, default="preds_test.csv",
                        help="Where to write per-image predictions CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms / dataset / loader (identical transform fn as training)
    test_tfms = build_transforms(img_size=args.img_size, train=False)
    test_ds = MRIDataset(args.test_image_dir, args.test_csv, transform=test_tfms)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Build model EXACTLY like training
    model = build_googlenet(
        num_classes=4,
        pretrained=not args.no_pretrained,
        aux_logits=not args.no_aux_logits
    ).to(device)

    # Load checkpoint (handles both dict-with-keys and raw state_dict)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)  # tolerate aux head toggles
    if missing or unexpected:
        print(f"[load_state_dict] missing keys: {missing}")
        print(f"[load_state_dict] unexpected keys: {unexpected}")

    # Inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device, num_classes=4)

    # Metrics
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    acc = correct / total if total > 0 else 0.0
    cm = confusion_matrix(y_true, y_pred, num_classes=4)
    prf = per_class_prf(cm)

    print(f"\n=== Test Results ===")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    for row in cm:
        print("  " + " ".join(f"{v:5d}" for v in row))
    print("\nPer-class (precision, recall, f1):")
    for i, (prec, rec, f1) in enumerate(prf):
        print(f"  class {i}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Save predictions CSV
    try:
        img_paths = read_csv_image_paths(args.test_csv)
        if len(img_paths) != total:
            print(f"[warn] #paths from CSV ({len(img_paths)}) != #preds ({total}); "
                  f"still writing with min length alignment.")
        n = min(len(img_paths), total)
        out_p = Path(args.save_preds)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "true", "pred"] + [f"prob_{i}" for i in range(4)]
            writer.writerow(header)
            for i in range(n):
                writer.writerow([img_paths[i], y_true[i], y_pred[i]] + [f"{p:.6f}" for p in y_prob[i]])
        print(f"\nSaved predictions to: {out_p.resolve()}")
    except Exception as e:
        print(f"[warn] Failed to write predictions CSV: {e}")


if __name__ == "__main__":
    main()
