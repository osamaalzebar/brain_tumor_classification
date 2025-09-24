# test_squeezenet_brain_tumor.py
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from dataset_squeezenet import MRIBrainTumorCSV, build_transforms, CLASS_NAMES


def build_squeezenet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Build SqueezeNet 1.1 with final conv replaced to output `num_classes`.
    This mirrors the training script's model head.
    """
    weights = SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = squeezenet1_1(weights=weights)
    in_channels = model.classifier[1].in_channels  # usually 512
    model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[List[float]]]:
    model.eval()
    all_t: List[int] = []
    all_p: List[int] = []
    all_pb: List[List[float]] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_t.extend(targets.tolist())
        all_p.extend(preds.tolist())
        all_pb.extend(probs.cpu().tolist())

    return all_t, all_p, all_pb


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
    parser = argparse.ArgumentParser(description="Test fine-tuned SqueezeNet on 4-class brain tumor dataset")
    parser.add_argument("--test_dir", type=str,default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/data",
                        help="Root folder for TEST images (filenames referenced in CSV)")
    parser.add_argument("--test_csv", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/image_labels_test.csv",
                        help="TEST CSV path with header: Image_path,label (labels 1..4)")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/squeezenet_best.pth",
                        help="Checkpoint saved by training script (.pth or .pt)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable ImageNet init (state_dict still loads)")
    parser.add_argument("--save_preds", type=str, default="preds_squeezenet_test.csv",
                        help="Output CSV of predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / loader (uses the exact normalization you put in dataset_squeezenet.py)
    test_tfms = build_transforms(args.img_size, train=False)
    test_ds = MRIBrainTumorCSV(args.test_dir, args.test_csv, transform=test_tfms)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model (exact same head replacement as training)
    model = build_squeezenet(num_classes=4, pretrained=not args.no_pretrained).to(device)

    # Load checkpoint (works whether file is a state_dict or {"model_state": ...})
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

    print("\n=== Test Results (SqueezeNet) ===")
    print(f"Samples: {total}")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    for row in cm:
        print("  " + " ".join(f"{v:5d}" for v in row))

    print("\nPer-class (precision, recall, f1):")
    for i, (prec, rec, f1) in enumerate(prf):
        cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i} ({cname}): P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Save per-image predictions
    try:
        img_paths = read_csv_image_paths(args.test_csv)  # original names from CSV
        n = min(len(img_paths), total)
        out_p = Path(args.save_preds)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["image_path", "true_idx(0-3)", "pred_idx(0-3)", "true_label(1-4)", "pred_label(1-4)"] \
                     + [f"prob_{i}" for i in range(4)]
            writer.writerow(header)
            for i in range(n):
                # Our dataset maps labels 1..4 -> 0..3, so for reporting 1..4 add +1 back:
                writer.writerow([
                    img_paths[i],
                    y_true[i],
                    y_pred[i],
                    y_true[i] + 1,
                    y_pred[i] + 1,
                    *[f"{p:.6f}" for p in y_prob[i]]
                ])
        print(f"\nSaved predictions to: {out_p.resolve()}")
    except Exception as e:
        print(f"[warn] Could not save predictions CSV: {e}")


if __name__ == "__main__":
    main()
