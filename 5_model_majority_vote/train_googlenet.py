# train_googlenet_brain_tumor.py
import os
import argparse
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights

from dataset_densenet import MRIDataset, build_transforms  # your existing dataset/transforms

# --------- Utilities ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def save_checkpoint(state, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)

# --------- Model builder ----------
def build_googlenet(num_classes: int = 4, pretrained: bool = True, aux_logits: bool = True):
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights, aux_logits=aux_logits)

    # Replace the primary classifier (PyTorch name for "loss3-classifier")
    model.fc = nn.Linear(1024, num_classes)

    # Replace auxiliary classifiers
    if aux_logits:
        model.aux1.fc = nn.Linear(1024, num_classes)
        model.aux2.fc = nn.Linear(1024, num_classes)

    return model

# --------- Training / Evaluation ----------
def train_one_epoch(model, loader, optimizer, scaler, device, criterion, aux_weight=0.3):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits, aux2_logits, aux1_logits = outputs  # main, aux2, aux1
                loss_main = criterion(logits, targets)
                loss_aux1 = criterion(aux1_logits, targets)
                loss_aux2 = criterion(aux2_logits, targets)
                loss = loss_main + aux_weight * (loss_aux1 + loss_aux2)
            else:
                logits = outputs
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bsz = images.size(0)
        running_loss += loss.item() * bsz
        running_acc += accuracy(logits, targets) * bsz
        n += bsz

    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterion(logits, targets)

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_acc += accuracy(logits, targets) * bsz
        n += bsz

    return total_loss / n, total_acc / n

# --------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune GoogLeNet on 4-class brain tumor MRI")
    # Separate roots + CSVs
    parser.add_argument("--train_image_dir", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/data', help="Root folder for TRAIN images")
    parser.add_argument("--train_csv", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/image_labels.csv', help="TRAIN CSV (Image_path,label)")
    parser.add_argument("--val_image_dir", type=str,default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/data', help="Root folder for VAL images")
    parser.add_argument("--val_csv", type=str, default=  '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/image_labels.csv', help="VAL CSV (Image_path,label)")


    # Training config
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default= 5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze all conv layers; train only classifier heads")
    parser.add_argument("--aux_weight", type=float, default=0.3,
                        help="Weight for each aux loss (aux1 and aux2)")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet initialization")
    parser.add_argument("--save_name", type=str, default="googlenet_best.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets / loaders
    train_tfms = build_transforms(img_size=args.img_size, train=True)
    val_tfms   = build_transforms(img_size=args.img_size, train=False)

    train_ds = MRIDataset(args.train_image_dir, args.train_csv, transform=train_tfms)
    val_ds   = MRIDataset(args.val_image_dir, args.val_csv, transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Model
    model = build_googlenet(
        num_classes=4,
        pretrained=not args.no_pretrained,
        aux_logits=True
    )

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = False
        for head in [model.fc, model.aux1.fc, model.aux2.fc]:
            for p in head.parameters():
                p.requires_grad = True

    model.to(device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss()  # raw logits → CE (no Softmax in model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Train
    best_acc = -math.inf
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / args.save_name

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion, aux_weight=args.aux_weight
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={dt:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, best_path)
            print(f"  ✓ Saved new best to: {best_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_acc:.4f} at {best_path}")

    # Example: produce "prob" (softmax) & "output" (argmax) for a mini-batch
    if len(val_loader) > 0:
        model.eval()
        with torch.no_grad():
            images, _ = next(iter(val_loader))
            images = images.to(device)
            outputs = model(images)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            prob = torch.softmax(logits, dim=1)   # "prob" layer analogue
            output = torch.argmax(prob, dim=1)    # "classification output"
            print("Example probs shape:", prob.shape)
            print("Example preds:", output[:8].tolist())

if __name__ == "__main__":
    main()
