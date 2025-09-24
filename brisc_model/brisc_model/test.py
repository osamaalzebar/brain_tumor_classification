#!/usr/bin/env python3
"""
Test/Evaluate a Swin Transformer classifier trained on BRISC brain tumor dataset.

- Loads a checkpoint produced by train_swin_brain_tumor.py (best_model.pth or last_model.pth)
- Evaluates on a CSV-indexed test set and reports:
  * Overall accuracy
  * Per-class precision / recall / F1
  * Confusion matrix (printed)
  * Saves detailed predictions to predictions.csv

CSV format header: "Image_path, label" with labels 1..4
(1=meningioma, 2=glioma, 3=pituitary, 4=no tumor).

Example:
python test_swin_brain_tumor.py \
  --data-root "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/data" \
  --csv       "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/Image_labels.csv" \
  --checkpoint "./checkpoints_swin_brisc/best_model.pth" \
  --outdir     "./test_results"

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import timm
except ImportError as e:
    raise SystemExit("This script requires the 'timm' package. Install via: pip install timm")


CLASS_NAMES = ['meningioma', 'glioma', 'pituitary', 'no_tumor']
NUM_CLASSES = 4


class CSVImageDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'image_path' in df.columns:
            img_col = 'image_path'
        elif 'image' in df.columns:
            img_col = 'image'
        else:
            raise ValueError("CSV must contain a column named 'Image_path' (case-insensitive).")
        if 'label' not in df.columns:
            raise ValueError("CSV must contain a column named 'label'.")

        self.records = []
        for _, row in df.iterrows():
            img_name = str(row[img_col]).strip()
            label = int(row['label'])
            if label not in (1, 2, 3, 4):
                raise ValueError(f"Label must be in {1,2,3,4}, got {label} for {img_name}")
            label_idx = label - 1
            p = Path(img_name)
            if not p.is_absolute():
                p = self.root / img_name
            if not p.exists():
                candidates = list(self.root.rglob(img_name))
                if candidates:
                    p = candidates[0]
                else:
                    raise FileNotFoundError(f"Image file not found: {p}")
            self.records.append((str(p), label_idx))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, path


def build_model(model_name: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    chkpt = torch.load(checkpoint_path, map_location='cpu')
    args_from_train = chkpt.get('args', {})
    # Prefer the trained model name if present in checkpoint
    trained_model_name = args_from_train.get('model', model_name)

    model = timm.create_model(trained_model_name, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(chkpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm: np.ndarray):
    # cm: [num_classes, num_classes] where rows=true, cols=pred
    precisions, recalls, f1s = [], [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return np.array(precisions), np.array(recalls), np.array(f1s)


def main():
    parser = argparse.ArgumentParser(description="Test Swin Transformer on BRISC brain tumor classification")
    parser.add_argument('--data-root', type=str, required=False, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/bangladesh_data/Raw")
    parser.add_argument('--csv', type=str, required=False, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/bangladesh_data/Raw/Image_labels.csv")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_swin_brisc/best_model.pth', help='Path to best_model.pth or last_model.pth from training')
    parser.add_argument('--model', type=str, default='swin_tiny_patch4_window7_224', help='Model name used during training (auto-read from checkpoint if available)')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--outdir', type=str, default='./test_results')

    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tfms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = CSVImageDataset(args.data_root, args.csv, transform=tfms)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, args.checkpoint, device)

    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in dl:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax_logits(logits).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_paths.extend(paths)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds, NUM_CLASSES)
    prec, rec, f1 = precision_recall_f1(cm)

    print("\nOverall Accuracy: {:.4f}".format(acc))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class metrics:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} ({name}): Precision={prec[i]:.4f}  Recall={rec[i]:.4f}  F1={f1[i]:.4f}")

    # Save predictions CSV
    pred_df = pd.DataFrame({
        'image_path': all_paths,
        'label_true_idx': all_labels,
        'label_true_name': [CLASS_NAMES[i] for i in all_labels],
        'label_pred_idx': all_preds,
        'label_pred_name': [CLASS_NAMES[i] for i in all_preds],
        'prob_meningioma': all_probs[:, 0],
        'prob_glioma': all_probs[:, 1],
        'prob_pituitary': all_probs[:, 2],
        'prob_no_tumor': all_probs[:, 3],
    })
    out_csv = Path(args.outdir) / 'predictions.csv'
    pred_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed predictions to: {out_csv}")

    # Save a summary report
    summary = {
        'accuracy': float(acc),
        'per_class_precision': {CLASS_NAMES[i]: float(prec[i]) for i in range(NUM_CLASSES)},
        'per_class_recall': {CLASS_NAMES[i]: float(rec[i]) for i in range(NUM_CLASSES)},
        'per_class_f1': {CLASS_NAMES[i]: float(f1[i]) for i in range(NUM_CLASSES)},
        'confusion_matrix': cm.tolist(),
    }
    summary_path = Path(args.outdir) / 'summary.json'
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == '__main__':
    main()
