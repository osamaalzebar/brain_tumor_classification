#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import csv
import random
from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from torchvision import transforms

# -----------------------
# Globals & utilities
# -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# -----------------------
# Transforms (as requested)
# -----------------------
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
# -----------------------
class BrainTumorCSVDataset(Dataset):
    """
    CSV format:
      Image_path,label
      img_001.png,1
      ...
    Labels: 1=meningioma, 2=glioma, 3=pituitary, 4=no_tumor
    """
    def __init__(self, root_dir: str, csv_path: str, transform: Callable):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames, \
                "CSV must contain headers: Image_path,label"
            for row in reader:
                rel = row["Image_path"].strip()
                y = int(row["label"]) - 1  # to 0..3
                # make full path robustly
                cand = rel if os.path.isabs(rel) else os.path.join(root_dir, rel)
                self.samples.append((cand, y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")  # MRI may be grayscale; convert to 3-ch for AlexNet norm
        img = self.transform(img)
        return img, y

# -----------------------
# Hybrid Attention (HAM)
# -----------------------
class ChannelAttention(nn.Module):
    """
    Channel attention with an adaptive mechanism (avg/max + weighted fusion)
    followed by a 1D conv across channels (ECA-style).
    """
    def __init__(self, channels: int, k_size: int | None = None, gamma: int = 2, b: int = 1):
        super().__init__()
        if k_size is None:
            # ECA kernel size heuristic -> ensure odd
            k = int(abs((math.log2(channels) / gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k_size = max(3, k)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg = F.adaptive_avg_pool2d(x, 1)  # (B,C,1,1)
        mx  = F.adaptive_max_pool2d(x, 1)
        # Adaptive fusion (roughly following the diagram idea)
        fused = self.alpha * avg + self.beta * mx + 0.5 * (avg * mx)
        y = fused.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B,1,C)
        y = self.conv1d(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return x * y.expand_as(x)

class SpatialAttentionDual(nn.Module):
    """
    Dual-branch spatial attention: split channels into two groups and apply
    a shared spatial-attention operator to each branch (CBAM-style).
    """
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # Shared conv across both branches
        self.shared = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def spatial_mask(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, Cx, H, W)
        avg_out = torch.mean(feat, dim=1, keepdim=True)    # (B,1,H,W)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)  # (B,1,H,W)
        x = torch.cat([avg_out, max_out], dim=1)           # (B,2,H,W)
        x = self.shared(x)                                 # (B,1,H,W)
        return self.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        C1 = C // 2
        x1 = x[:, :C1, :, :]
        x2 = x[:, C1:, :, :]

        m1 = self.spatial_mask(x1)
        m2 = self.spatial_mask(x2)

        x1 = x1 * m1
        x2 = x2 * m2
        return torch.cat([x1, x2], dim=1)

class HybridAttentionModule(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttentionDual(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

# -----------------------
# SAlexNet2 backbone (per your diagram; AlexNet-ish layout with HAMs)
# -----------------------
class SAlexNet2(nn.Module):
    """
    A light AlexNet-style model with 5 conv stages and HAM insertions:
      - Stage1: multiple 3x3 convs (to 64), HAM, MaxPool
      - Stage2: 5x5 conv (to 192), HAM, MaxPool
      - Stage3: 3x3 conv (to 384), HAM
      - Stage4: 3x3 conv (to 256), HAM
      - Stage5: 3x3 conv (to 256), HAM, MaxPool
      - AdaptiveAvgPool -> (6x6) -> FC
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # ---- Stage 1 (output: 64)
        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.ham1 = HybridAttentionModule(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Stage 2 (output: 192)
        self.s2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )
        self.ham2 = HybridAttentionModule(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Stage 3 (output: 384)
        self.s3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
        )
        self.ham3 = HybridAttentionModule(384)

        # ---- Stage 4 (output: 256)
        self.s4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.ham4 = HybridAttentionModule(256)

        # ---- Stage 5 (output: 256)
        self.s5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.ham5 = HybridAttentionModule(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # -> 256 x 6 x 6 (9216)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

        # Kaiming init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s1(x)
        x = self.ham1(x)
        x = self.pool1(x)

        x = self.s2(x)
        x = self.ham2(x)
        x = self.pool2(x)

        x = self.s3(x)
        x = self.ham3(x)

        x = self.s4(x)
        x = self.ham4(x)

        x = self.s5(x)
        x = self.ham5(x)
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)  # logits
        return x

# -----------------------
# Train / validate loops
# -----------------------
@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / max(1, total), running_loss / max(1, total)

def train(
    train_root: str,
    train_csv: str,
    val_root: str,
    val_csv: str,
    out_dir: str = "./checkpoints",
    img_size: int = 224,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    num_workers: int = 4,
    seed: int = 42
):
    seed_everything(seed)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_tf = build_transforms(img_size, train=True)
    val_tf   = build_transforms(img_size, train=False)

    train_ds = BrainTumorCSVDataset(train_root, train_csv, transform=train_tf)
    val_ds   = BrainTumorCSVDataset(val_root,   val_csv,   transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    # Model, loss, optimizer, scheduler
    model = SAlexNet2(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_path = os.path.join(out_dir, "salexnet2_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, total)
        val_acc, val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{epochs:03d} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "args": {
                    "img_size": img_size,
                    "classes": CLASS_NAMES
                }
            }, best_path)
            print(f"  🔥 New best model saved to: {best_path} (val_acc={best_acc:.4f})")

    print(f"Training done. Best val_acc = {best_acc:.4f}")
    return best_path

# -----------------------
# Entry
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SAlexNet2 with HAM on brain tumor dataset")
    parser.add_argument("--train_root", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train', help="Folder containing training images")
    parser.add_argument("--train_csv",  type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv', help="CSV with Image_path,label for training")
    parser.add_argument("--val_root",   type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val', help="Folder containing validation images")
    parser.add_argument("--val_csv",    type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv', help="CSV with Image_path,label for validation")
    parser.add_argument("--out_dir",    type=str, default="./checkpoints")
    parser.add_argument("--img_size",   type=int, default=224)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    train(
        train_root=args.train_root,
        train_csv=args.train_csv,
        val_root=args.val_root,
        val_csv=args.val_csv,
        out_dir=args.out_dir,
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )

