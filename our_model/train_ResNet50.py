#!/usr/bin/env python3
# train_resnet50_cls.py
import os
import argparse
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset_vgg16 import BrainMRIDataset


def build_model(num_classes: int = 4, freeze_until: str = "none"):
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Optionally freeze layers
    if freeze_until == "features":
        for name, param in model.named_parameters():
            if not name.startswith("fc"):  # freeze all but classifier
                param.requires_grad = False
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, running_loss = 0, 0, 0.0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (make sure BrainMRIDataset uses ResNet normalization: mean/std below)
    train_ds = BrainMRIDataset(args.train_root, args.train_csv, img_size=args.image_size, train=True)
    val_ds   = BrainMRIDataset(args.val_root,   args.val_csv,   img_size=args.image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=4, freeze_until=args.freeze).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc  = correct / max(1, total)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                args.out,
            )

    print(f"Training finished. Best val acc: {best_acc:.4f}. Saved to: {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ResNet50 on brain MRI with separate train/val sets.")
    p.add_argument("--train-root", type=str, required=True)
    p.add_argument("--train-csv",  type=str, required=True)
    p.add_argument("--val-root",   type=str, required=True)
    p.add_argument("--val-csv",    type=str, required=True)
    p.add_argument("--out", type=str, default="best_resnet50_brain_cls.pth")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--freeze", type=str, default="none", choices=["none", "features"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    train(args)
