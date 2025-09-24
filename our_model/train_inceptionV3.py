#!/usr/bin/env python3
# train_inceptionv3_concat.py
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import models

from dataset_inceptionV3 import BrainCSVSet, IMAGENET_MEAN, IMAGENET_STD

# -------------------------
# Model: InceptionV3 taps + concat head
# -------------------------
class InceptionV3ConcatHead(nn.Module):
    """
    Tap features from three Inception blocks:
      - Mixed_6e
      - Mixed_7a
      - Mixed_7c
    Each tap -> GAP -> Dropout -> FC(256) -> ReLU
    Concatenate (256*3=768) -> Linear(4)
    """
    def __init__(self, num_classes: int = 4, dropout_p: float = 0.4, freeze_until: str = "Mixed_5d"):
        super().__init__()
        # Load pretrained inception_v3
        self.base = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        # Throw away the original classifier
        self.base.fc = nn.Identity()

        # Optionally freeze early layers up to a named block
        self._freeze_stages(until=freeze_until)

        # Feature taps captured via forward hooks
        self._taps: Dict[str, torch.Tensor] = {}

        # Register hooks on the target blocks
        self._register_tap("Mixed_6e", self.base.Mixed_6e)
        self._register_tap("Mixed_7a", self.base.Mixed_7a)
        self._register_tap("Mixed_7c", self.base.Mixed_7c)

        # Heads for each tap (channels depend on InceptionV3 internals)
        # Torchvision InceptionV3:
        #   Mixed_6e out channels = 768
        #   Mixed_7a out channels = 1280
        #   Mixed_7c out channels = 2048
        self.pool = nn.AdaptiveAvgPool2d(1)

        def head(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_ch, 256),
                nn.ReLU(inplace=True),
            )

        self.head_6e = head(768)
        self.head_7a = head(1280)
        self.head_7c = head(2048)

        self.classifier = nn.Linear(256 * 3, num_classes)

    # --- utils ---
    def _register_tap(self, name: str, module: nn.Module):
        def hook(_, __, output):
            self._taps[name] = output
        module.register_forward_hook(hook)

    def _freeze_stages(self, until: str):
        """
        Freeze parameters up to (and including) a certain stage name. 
        Pass until=None to keep all trainable.
        """
        freeze = True if until else False
        for name, p in self.base.named_parameters():
            if freeze:
                p.requires_grad = False
            if until and until in name:
                # after we pass this module name once, unfreeze the rest
                freeze = False

    # --- forward ---
    def forward(self, x):
        # Forward through base; taps get captured by hooks
        _ = self.base(x)

        # Expect the taps to be present
        feat_6e = self._taps.get("Mixed_6e", None)
        feat_7a = self._taps.get("Mixed_7a", None)
        feat_7c = self._taps.get("Mixed_7c", None)

        if feat_6e is None or feat_7a is None or feat_7c is None:
            raise RuntimeError("Feature taps were not captured. Check hook registration.")

        # GAP -> flatten
        z6e = self.pool(feat_6e).squeeze(-1).squeeze(-1)   # [B, 768]
        z7a = self.pool(feat_7a).squeeze(-1).squeeze(-1)   # [B, 1280]
        z7c = self.pool(feat_7c).squeeze(-1).squeeze(-1)   # [B, 2048]

        z6e = self.head_6e(z6e)    # [B, 256]
        z7a = self.head_7a(z7a)    # [B, 256]
        z7c = self.head_7c(z7c)    # [B, 256]

        z = torch.cat([z6e, z7a, z7c], dim=1)  # [B, 768]
        logits = self.classifier(z)             # [B, 4]
        return logits

# -------------------------
# Train / Eval
# -------------------------
def accuracy_top1(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None):
    model.train()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_top1(logits, labels) * bs
        n += bs

    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_top1(logits, labels) * bs
        n += bs

    return running_loss / n, running_acc / n

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune InceptionV3 with feature-tap concatenation (4-class).")
    ap.add_argument("--train-root", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/inceptionV3_brisc /classification_task/train/data")
    ap.add_argument("--train-csv",  type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/inceptionV3_brisc /classification_task/train/image_labels.csv")
    ap.add_argument("--val-root",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/inceptionV3_brisc /classification_task/val/data")
    ap.add_argument("--val-csv",    type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/inceptionV3_brisc /classification_task/val/image_labels.csv")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default= 5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--freeze-until", type=str, default="Mixed_5d", help="Freeze params up to this block name (incl). Set '' to train all.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="./checkpoint")
    ap.add_argument("--use-amp", action="store_true", help="Enable mixed precision (AMP)")

    args = ap.parse_args()
    device = torch.device(args.device)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Datasets / Loaders (299 for Inception)
    train_ds = BrainCSVSet(args.train_root, args.train_csv, train=True,  img_size=299)
    val_ds   = BrainCSVSet(args.val_root,   args.val_csv,   train=False, img_size=299)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Model
    model = InceptionV3ConcatHead(num_classes=4, dropout_p=args.dropout, freeze_until=(args.freeze_until or None)).to(device)

    # Optimizer / Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == "cuda") else None

    best_acc = 0.0
    best_path = Path(args.outdir) / "best_incv3_concat.pth"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, device, scaler, criterion)
        va_loss, va_acc = evaluate(model, val_dl, device, criterion)

        print(f"Epoch {epoch:03d} | "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.4f}")

        # Save best
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": best_acc,
                "args": vars(args),
            }, best_path)
            print(f"  -> saved best to {best_path} (val_acc={best_acc:.4f})")

    print(f"Training done. Best val acc: {best_acc:.4f}. Best model: {best_path}")

if __name__ == "__main__":
    main()

