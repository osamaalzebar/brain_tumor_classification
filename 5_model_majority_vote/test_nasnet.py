import argparse
import csv
import os
from typing import Tuple, List
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Optional: per-class report & confusion matrix
_HAS_SK = True
try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    _HAS_SK = False

import pretrainedmodels


# ---------------------------
# Dataset + Transforms
# ---------------------------

class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        # not used in test; kept for parity
        return img

def build_transforms(
    img_size: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class BrainTumorCSVDataset(Dataset):
    """
    CSV header: Image_path,label
      - Image_path is filename inside images_dir
      - label in {1,2,3,4} -> mapped to {0,1,2,3}
    """
    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self._load()

    def _load(self):
        if not os.path.isdir(self.images_dir):
            raise NotADirectoryError(self.images_dir)
        if not os.path.isfile(self.labels_csv):
            raise FileNotFoundError(self.labels_csv)

        with open(self.labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames, \
                "CSV must have header: Image_path,label"

            for row in reader:
                fn = row["Image_path"].strip()
                label_raw = int(row["label"])
                label = label_raw - 1  # map 1..4 -> 0..3
                path = os.path.join(self.images_dir, fn)
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                self.samples.append((fn, path, label))

        if len(self.samples) == 0:
            raise ValueError("No samples found")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        fn, path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return fn, img, torch.tensor(label, dtype=torch.long)


# ---------------------------
# Model wrapper
# ---------------------------

class NASNetMobileClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 4):
        super().__init__()
        self.backbone = backbone
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_probs: bool = False):
        logits = self.backbone(x)
        if return_probs:
            probs = self.softmax(logits)
            preds = torch.argmax(probs, dim=1)
            return logits, probs, preds
        return logits


def build_nasnet_mobile(num_classes: int = 4):
    """
    Build the same architecture used in training:
    - pretrainedmodels nasnetamobile backbone (we won't download weights here;
      we load your fine-tuned checkpoint instead).
    - replace last_linear with 4-class head.
    - get mean/std/img_size from pretrained settings for transforms.
    """
    base = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained=None)
    in_features = base.last_linear.in_features
    base.last_linear = nn.Linear(in_features, num_classes)

    setting = pretrainedmodels.pretrained_settings['nasnetamobile']['imagenet']
    mean = tuple(setting['mean'])
    std = tuple(setting['std'])
    img_size = setting.get('input_size', (3, 224, 224))[1]

    model = NASNetMobileClassifier(base, num_classes=num_classes)
    return model, mean, std, img_size


# ---------------------------
# Evaluation
# ---------------------------

CLASSES = ["meningioma", "glioma", "pituitary", "no_tumor"]

@torch.no_grad()
def evaluate(model, loader, device, save_predictions_csv: str = "", save_report_csv: str = ""):
    model.eval()
    y_true = []
    y_pred = []
    pred_rows = []  # optional per-sample predictions

    for fns, imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, probs, preds = model(imgs, return_probs=True)

        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if save_predictions_csv:
            probs_np = probs.cpu().numpy()
            for i, fn in enumerate(fns):
                row = {
                    "Image_path": fn,
                    "true_label_idx": int(targets[i].cpu().item()),
                    "pred_label_idx": int(preds[i].cpu().item()),
                    "pred_label_name": CLASSES[int(preds[i].cpu().item())],
                }
                # add class probs
                for ci, cname in enumerate(CLASSES):
                    row[f"prob_{cname}"] = float(probs_np[i, ci])
                pred_rows.append(row)

    # Overall accuracy
    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Confusion matrix & per-class report
    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(cm)

        report = classification_report(
            y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0
        )
        print("\nClassification Report:")
        print(report)

        # Save per-class metrics to CSV if requested
        if save_report_csv:
            # Recompute as dict for CSV
            rep_dict = classification_report(
                y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
            )
            with open(save_report_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["label", "precision", "recall", "f1-score", "support"])
                for k in CLASSES + ["accuracy", "macro avg", "weighted avg"]:
                    if k in rep_dict:
                        v = rep_dict[k]
                        if k == "accuracy":
                            w.writerow([k, "", "", f"{v:.6f}", ""])
                        else:
                            w.writerow([
                                k,
                                f"{v.get('precision', 0):.6f}",
                                f"{v.get('recall', 0):.6f}",
                                f"{v.get('f1-score', 0):.6f}",
                                f"{v.get('support', 0)}",
                            ])
            print(f"[INFO] Saved classification report CSV to: {save_report_csv}")
    else:
        print("(scikit-learn not installed; skipping confusion matrix / per-class report)")
        print("Install with: pip install scikit-learn")

    # Save per-sample predictions if requested
    if save_predictions_csv and len(pred_rows) > 0:
        with open(save_predictions_csv, "w", newline="") as f:
            fieldnames = list(pred_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in pred_rows:
                w.writerow(r)
        print(f"[INFO] Saved per-sample predictions to: {save_predictions_csv}")


def main():
    parser = argparse.ArgumentParser("Test NASNet-Mobile on brain tumor MRI")
    parser.add_argument("--test_images", type=str,
        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/data")
    parser.add_argument("--test_csv", type=str,
        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/image_labels_test.csv")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_nasnet_mobile.pth",
        help="Path to fine-tuned checkpoint (best_nasnet_mobile.pth)")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--save_predictions_csv", type=str, default="./predictions_test.csv",
        help="Where to save per-sample predictions (CSV). Empty to skip.")
    parser.add_argument("--save_report_csv", type=str, default="./classification_report_test.csv",
        help="Where to save per-class metrics (CSV). Empty to skip.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (same architecture as training) + transforms
    model, mean, std, img_size = build_nasnet_mobile(num_classes=4)
    model = model.to(device)

    # Load checkpoint (state dict from training)
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)  # supports either {"model":...} or raw state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[INFO] Missing keys while loading (first few): {missing[:6]}{'...' if len(missing)>6 else ''}")
    if unexpected:
        print(f"[INFO] Unexpected keys while loading (first few): {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")

    # Data
    test_tfms = build_transforms(img_size=img_size, mean=mean, std=std)
    test_ds = BrainTumorCSVDataset(args.test_images, args.test_csv, transform=test_tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True,
                             collate_fn=lambda batch: (
                                 [b[0] for b in batch],      # filenames
                                 torch.stack([b[1] for b in batch], dim=0),
                                 torch.stack([b[2] for b in batch], dim=0),
                             ))

    # Eval
    evaluate(
        model, test_loader, device,
        save_predictions_csv=args.save_predictions_csv or "",
        save_report_csv=args.save_report_csv or "",
    )


if __name__ == "__main__":
    main()
