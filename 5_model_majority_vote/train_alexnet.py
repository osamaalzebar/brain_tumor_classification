# train_alexnet_brain_tumor.py
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights

from dataset_alexnet import MRIBrainTumorCSV, build_transforms, CLASS_NAMES


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_alexnet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    AlexNet with last THREE FC layers replaced:
      fc6: 9216->4096, fc7: 4096->4096, fc8: 4096->num_classes
    """
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)

    in_features = model.classifier[1].in_features  # usually 9216
    hidden = 4096
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes),
    )
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None):
    model.train()
    total_loss = total_acc = n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits.detach(), targets) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    model.eval()
    total_loss = total_acc = n = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, total_acc / n


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AlexNet on 4-class brain tumor MRI")
    parser.add_argument("--train_dir", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/data")
    parser.add_argument("--train_csv", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/image_labels.csv")
    parser.add_argument("--val_dir",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/data")
    parser.add_argument("--val_csv",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/image_labels.csv")

    parser.add_argument("--img_size", type=int, default=224)  # use 227 if you prefer the original AlexNet size
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5, help="Base LR for transferred layers")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="alexnet_best.pth")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_tfms = build_transforms(args.img_size, train=True)
    val_tfms   = build_transforms(args.img_size, train=False)
    train_ds = MRIBrainTumorCSV(args.train_dir, args.train_csv, transform=train_tfms)
    val_ds   = MRIBrainTumorCSV(args.val_dir, args.val_csv, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model
    model = build_alexnet(num_classes=4, pretrained=not args.no_pretrained).to(device)

    # Give the NEW FC layers a higher LR (10x) than transferred layers
    fc1, fc2, fc3 = model.classifier[1], model.classifier[4], model.classifier[6]
    base_params = []
    new_params_w, new_params_b = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p is fc1.weight or p is fc2.weight or p is fc3.weight:
            new_params_w.append(p)
        elif (fc1.bias is not None and p is fc1.bias) or (fc2.bias is not None and p is fc2.bias) or (fc3.bias is not None and p is fc3.bias):
            new_params_b.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.Adam([
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": new_params_w, "lr": args.lr * 10.0, "weight_decay": args.weight_decay},
        {"params": new_params_b, "lr": args.lr * 10.0, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Train
    best_acc, best_path = 0.0, None
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_dir) / args.save_name

    print(f"Classes: {CLASS_NAMES}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        va_loss, va_acc = validate(model, val_loader, device, criterion)
        scheduler.step()
        dt = time.time() - t0

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "best_val_acc": best_acc,
                "epoch": epoch,
                "args": vars(args),
                "class_names": CLASS_NAMES,
            }, save_path)
            best_path = save_path

    print("\nTraining complete.")
    if best_path:
        print(f"Best val_acc = {best_acc:.4f} | Saved: {best_path.resolve()}")
    else:
        print("No improvement recorded; model not saved.")


if __name__ == "__main__":
    main()
