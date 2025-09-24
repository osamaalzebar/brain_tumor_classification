#!/usr/bin/env python3
"""
Train a Swin Transformer classifier for brain tumor classification (BRISC)

Edits in this version:
- Added RandomRotate90() augmentation.
- Added build_transforms() that applies:
    * Resize -> RandomHorizontalFlip(0.5) -> RandomRotate90 -> ToTensor -> Normalize
      for training; and Resize -> ToTensor -> Normalize for validation.
- Uses ImageNet mean/std, suitable for Swin models (pretrained or from scratch).

Usage example:
python train_swin_brain_tumor.py \
  --train-root "/path/to/train/data_root" \
  --train-csv  "/path/to/train/Image_labels.csv" \
  --val-root   "/path/to/val/data_root" \
  --val-csv    "/path/to/val/Image_labels.csv" \
  --outdir     "./checkpoints_swin_brisc"
"""

import argparse
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import timm
except ImportError as e:
    raise SystemExit("This script requires the 'timm' package. Install via: pip install timm")


# ---------------------------
# Constants for normalization
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------
# Augmentations / Transforms
# ---------------------------
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


# ---------------------------
# Dataset
# ---------------------------
class CSVImageDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        df = pd.read_csv(csv_path)
        # Normalize column names (strip spaces, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        # Accept variants like 'image_path' or 'image' etc.
        if 'image_path' in df.columns:
            img_col = 'image_path'
        elif 'image' in df.columns:
            img_col = 'image'
        else:
            raise ValueError("CSV must contain a column named 'Image_path' (case-insensitive).")
        if 'label' not in df.columns:
            raise ValueError("CSV must contain a column named 'label'.")

        self.samples = []
        for _, row in df.iterrows():
            img_name = str(row[img_col]).strip()
            label = int(row['label'])
            # Convert labels 1..4 -> 0..3
            if label not in (1, 2, 3, 4):
                raise ValueError(f"Label must be in {1,2,3,4}, got {label} for {img_name}")
            label_idx = label - 1

            # Build absolute image path
            p = Path(img_name)
            if not p.is_absolute():
                p = self.root / img_name
            if not p.exists():
                # Try case-insensitive match inside root if not found
                candidates = list(self.root.rglob(img_name))
                if candidates:
                    p = candidates[0]
                else:
                    raise FileNotFoundError(f"Image file not found: {p}")

            self.samples.append((p, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Ensure RGB (Swin expects 3 channels)
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Utilities
# ---------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


# ---------------------------
# Training / Validation loops
# ---------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += accuracy(outputs.detach(), labels) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Swin Transformer on BRISC brain tumor classification")

    parser.add_argument('--train-root', type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train")
    parser.add_argument('--train-csv', type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv")
    parser.add_argument('--val-root', type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val")
    parser.add_argument('--val-csv', type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv")

    parser.add_argument('--outdir', type=str, default='./checkpoints_swin_brisc')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--model', type=str, default='swin_tiny_patch4_window7_224',
                        help='Any timm Swin variant, e.g., swin_tiny_patch4_window7_224, swin_small_patch4_window7_224')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained weights')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms with requested augmentation
    train_tfms = build_transforms(img_size=args.img_size, train=True)
    val_tfms = build_transforms(img_size=args.img_size, train=False)

    train_ds = CSVImageDataset(args.train_root, args.train_csv, transform=train_tfms)
    val_ds = CSVImageDataset(args.val_root, args.val_csv, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=4)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine scheduler with warmup (5 epochs)
    total_epochs = args.epochs
    warmup_epochs = min(5, total_epochs // 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine decay to 10% of base lr
        progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.1 + 0.9 * (0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    history = []

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print('-' * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        })

        print(f"Train   - loss: {train_loss:.4f} | acc: {train_acc*100:.2f}%")
        print(f"Val     - loss: {val_loss:.4f} | acc: {val_acc*100:.2f}%")
        print(f"LR      - {scheduler.get_last_lr()[0]:.6f}")

        # Save last
        last_path = Path(args.outdir) / 'last_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'history': history,
            'args': vars(args),
        }, last_path)

        # Save best on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = Path(args.outdir) / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'best_val_acc': best_val_acc,
                'history': history,
                'args': vars(args),
            }, best_path)
            print(f"** Saved new best checkpoint: {best_path} (val_acc={best_val_acc*100:.2f}%)")

    # Save history as CSV for later plotting
    hist_df = pd.DataFrame(history)
    hist_csv = Path(args.outdir) / 'training_history.csv'
    hist_df.to_csv(hist_csv, index=False)
    print(f"Training complete. Best val acc: {best_val_acc*100:.2f}%")
    print(f"Checkpoints in: {args.outdir}")


if __name__ == '__main__':
    main()
