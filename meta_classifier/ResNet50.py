# fine_tune_resnet50_brain_tumor.py
import os
import csv
import math
import random
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# -----------------------
# Config (edit paths if needed)
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

# Guessed val images root (adjust if different)
VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

OUTPUT_DIR     = "./outputs_resnet50"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "resnet50_best.pth")

NUM_CLASSES = 4
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 4
LR = 5e-5
NUM_WORKERS = 4

SEED = 42

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -----------------------
# Transforms (as requested)
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)  # 0 to 3 rotations
        return img.rotate(90 * k)

def build_transforms(img_size: int = 224, train: bool = True) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

# -----------------------
# Dataset
# CSV header: Image_path,label
# label mapping in CSV: 1=meningioma, 2=glioma, 3=pituitary, 4=no tumor
# CrossEntropyLoss expects class indices in [0..C-1] -> subtract 1
# -----------------------
class CsvImageDataset(Dataset):
    def __init__(self, img_root: str, csv_path: str, transform: Callable):
        self.img_root = img_root
        self.transform = transform

        df = pd.read_csv(csv_path)
        # Normalize column names (trim spaces)
        df.columns = [c.strip() for c in df.columns]
        if "Image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: 'Image_path,label'")

        # build list
        self.samples: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            img_name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            # Full path: if CSV has only filename, join with img_root
            # If CSV accidentally contains 'data/...' keep it robust:
            candidate = os.path.join(self.img_root, os.path.basename(img_name))
            self.samples.append((candidate, label_1_to_4 - 1))  # shift to 0..3

        # Optionally filter only existing files (safety)
        existing = [(p, y) for (p, y) in self.samples if os.path.isfile(p)]
        missing = len(self.samples) - len(existing)
        if missing > 0:
            print(f"Warning: {missing} images listed in CSV not found under {img_root}. They will be skipped.")
        self.samples = existing

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for csv={csv_path} with root={img_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# -----------------------
# Model
# -----------------------
def build_resnet50(num_classes: int = 4) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # logits
    return model

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

# -----------------------
# Train / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total

# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms = build_transforms(IMG_SIZE, train=True)
    val_tfms   = build_transforms(IMG_SIZE, train=False)

    train_ds = CsvImageDataset(TRAIN_IMG_ROOT, TRAIN_CSV, transform=train_tfms)
    val_ds   = CsvImageDataset(VAL_IMG_ROOT, VAL_CSV, transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = build_resnet50(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()  # expects class indices 0..3
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%   "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save best by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR
                }
            }, BEST_MODEL_PATH)
            print(f"  ↳ Saved best model (Val Acc: {best_val_acc*100:.2f}%) to {BEST_MODEL_PATH}")

    print("Training complete.")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")

    # Optional: show example softmax probabilities on a small val batch
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            print("Example softmax probabilities (first batch):")
            print(probs[:min(4, probs.size(0))].cpu())
            break

if __name__ == "__main__":
    main()
