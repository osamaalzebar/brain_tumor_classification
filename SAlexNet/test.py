#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
from typing import Callable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# -----------------------
# Constants
# -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]

# -----------------------
# Test-time transforms (NO augmentation)
# -----------------------
def build_test_transforms(img_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# -----------------------
# Dataset (same CSV structure as training)
# CSV headers: Image_path,label  with labels in {1,2,3,4}
# -----------------------
class BrainTumorCSVDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, transform: Callable):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int, str]] = []  # (abs_path, label0..3, rel_path)

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames, \
                "CSV must contain headers: Image_path,label"
            for row in reader:
                rel = row["Image_path"].strip()
                y = int(row["label"]) - 1  # map to 0..3
                full = rel if os.path.isabs(rel) else os.path.join(root_dir, rel)
                self.samples.append((full, y, rel))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, rel = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y, rel

# -----------------------
# Hybrid Attention blocks (same as training script)
# -----------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, k_size: int | None = None, gamma: int = 2, b: int = 1):
        super().__init__()
        if k_size is None:
            k = int(abs((math.log2(channels) / gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k_size = max(3, k)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(1.0))
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        fused = self.alpha * avg + self.beta * mx + 0.5 * (avg * mx)
        y = fused.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B,1,C)
        y = self.conv1d(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return x * y.expand_as(x)

class SpatialAttentionDual(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.shared = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def spatial_mask(self, feat: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.shared(x)
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
# SAlexNet2 (same as training)
# -----------------------
class SAlexNet2(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
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

        self.s2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )
        self.ham2 = HybridAttentionModule(192)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.s3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
        )
        self.ham3 = HybridAttentionModule(384)

        self.s4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.ham4 = HybridAttentionModule(256)

        self.s5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.ham5 = HybridAttentionModule(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
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
        x = self.s1(x); x = self.ham1(x); x = self.pool1(x)
        x = self.s2(x); x = self.ham2(x); x = self.pool2(x)
        x = self.s3(x); x = self.ham3(x)
        x = self.s4(x); x = self.ham4(x)
        x = self.s5(x); x = self.ham5(x); x = self.pool5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# -----------------------
# Evaluation
# -----------------------
@torch.no_grad()
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    running_loss = 0.0
    all_rows = []  # (rel_path, true1..4, pred1..4, prob)

    num_classes = 4
    cls_correct = [0] * num_classes
    cls_total   = [0] * num_classes

    for imgs, labels, rel in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        # per-class tallies
        for c in range(num_classes):
            mask_c = (labels == c)
            cls_total[c]   += int(mask_c.sum().item())
            cls_correct[c] += int(((preds == labels) & mask_c).sum().item())

        probs = logits.softmax(dim=1)
        for r, t, p, pr in zip(rel, labels.cpu().tolist(),
                               preds.cpu().tolist(), probs.cpu().tolist()):
            all_rows.append((r, t + 1, p + 1, float(max(pr))))

    overall_acc = correct / max(1, total)
    avg_loss = running_loss / max(1, total)

    per_class_acc = [
        (cls_correct[c] / cls_total[c]) if cls_total[c] > 0 else float("nan")
        for c in range(num_classes)
    ]
    # macro average over classes that appear
    valid = [a for a, ct in zip(per_class_acc, cls_total) if ct > 0]
    macro_acc = sum(valid) / max(1, len(valid))

    return overall_acc, avg_loss, all_rows, per_class_acc, macro_acc, cls_total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test SAlexNet2 on brain tumor dataset (no augmentation).")
    parser.add_argument("--data_root", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test', help="Folder containing test images")
    parser.add_argument("--csv", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/Image_labels.csv', help="CSV with Image_path,label for test set")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints/salexnet2_best.pt', help="Path to salexnet2_best.pt")
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_csv",   type=str, default="", help="Optional: write predictions to this CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & loader
    tfm = build_test_transforms(args.img_size)
    ds  = BrainTumorCSVDataset(args.data_root, args.csv, transform=tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Model
    model = SAlexNet2(num_classes=4).to(device)

    # Load checkpoint (as saved by the training script)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)  # allow raw state_dict too
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # Evaluate
    acc, loss, rows, pc_acc, macro_acc, cls_total = evaluate(model, loader, device)
    print(f"Test  | loss: {loss:.4f} | overall_acc: {acc:.4f} | macro_acc(4 classes): {macro_acc:.4f}")

    for i, a in enumerate(pc_acc):
        if not math.isnan(a):
            print(f"  Class {i + 1} ({CLASS_NAMES[i]}), n={cls_total[i]}: acc={a:.4f}")


if __name__ == "__main__":
    main()
