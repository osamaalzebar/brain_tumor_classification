# test_resnet50v2_custom_head.py
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm

# -----------------------
# Paths / Config (EDIT if needed)
# -----------------------
TEST_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/bangladesh_data/Raw/data"
TEST_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/bangladesh_data/Raw/Image_labels.csv"

CKPT_PATH     = "./outputs_resnet50v2_custom/resnet50v2_custom_best.pth"
OUTPUT_DIR    = "./outputs_resnet50v2_custom"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

# Optional: save per-image predictions
SAVE_PREDICTIONS_CSV = True
PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "test_predictions.csv")

CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]  # 0..3

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# -----------------------
# Transforms (eval only)
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------
# Dataset (CSV: Image_path,label; labels 1..4 -> 0..3)
# -----------------------
class CsvImageDataset(Dataset):
    def __init__(self, img_root: str, csv_path: str, transform):
        self.img_root = img_root
        self.transform = transform

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: 'Image_path,label'")

        samples: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            img_name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            path = os.path.join(self.img_root, os.path.basename(img_name))
            samples.append((path, label_1_to_4 - 1))

        self.samples = [(p, y) for (p, y) in samples if os.path.isfile(p)]
        missing = len(samples) - len(self.samples)
        if missing > 0:
            print(f"Warning: {missing} images in CSV not found under {self.img_root}; skipped.")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples for {csv_path} with root {img_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y, os.path.basename(path)

# -----------------------
# Model (must match training architecture)
# -----------------------
class ResNet50V2Custom(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = timm.create_model("resnetv2_50", pretrained=False, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features  # ~2048

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm1d(feat_dim)

        self.fc1 = nn.Linear(feat_dim, 1280)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(1280)

        self.classifier = nn.Linear(1280, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_logits: bool = False):
        feats = self.backbone.forward_features(x)      # [B, C, H, W]
        x = self.gap(feats).flatten(1)                 # [B, C]
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        logits = self.classifier(x)                    # [B, num_classes]
        if return_logits:
            return logits
        return self.softmax(logits)

def load_checkpoint(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

@torch.no_grad()
def per_class_accuracy(cm: np.ndarray):
    accs = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        accs.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return accs

# -----------------------
# Main (test)
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = CsvImageDataset(TEST_IMG_ROOT, TEST_CSV, transform=eval_tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    model = ResNet50V2Custom(NUM_CLASSES).to(device)
    ckpt = load_checkpoint(model, CKPT_PATH)
    model.eval()

    all_preds, all_labels = [], []
    all_names, all_probs = [], []

    with torch.no_grad():
        for imgs, labels, names in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs, return_logits=True)
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_names.extend(list(names))
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    probs  = np.concatenate(all_probs, axis=0)  # [N, 4]

    overall_acc = (y_pred == y_true).mean()
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    cls_acc = per_class_accuracy(cm)
    macro_acc = float(np.mean(cls_acc))

    print("\n=== ResNet50V2 Custom-Head: Test Results ===")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print("Per-class Accuracy:")
    for i, a in enumerate(cls_acc):
        cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i} ({cname}): {a*100:.2f}%")
    print(f"Macro-average Accuracy: {macro_acc*100:.2f}%")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    if SAVE_PREDICTIONS_CSV:
        df = pd.DataFrame({
            "image": all_names,
            "true_label_idx": y_true,
            "pred_label_idx": y_pred,
            "true_label_name": [CLASS_NAMES[i] for i in y_true],
            "pred_label_name": [CLASS_NAMES[i] for i in y_pred],
            "p_meningioma": probs[:, 0],
            "p_glioma":     probs[:, 1],
            "p_pituitary":  probs[:, 2],
            "p_no_tumor":   probs[:, 3],
        })
        df.to_csv(PREDICTIONS_CSV_PATH, index=False)
        print(f"\nSaved per-image predictions to: {PREDICTIONS_CSV_PATH}")

if __name__ == "__main__":
    main()
