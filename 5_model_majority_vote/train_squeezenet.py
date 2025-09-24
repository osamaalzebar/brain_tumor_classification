# train_squeezenet_brain_tumor.py
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from dataset_squeezenet import MRIBrainTumorCSV, build_transforms, CLASS_NAMES


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_squeezenet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Replace final conv ("conv10" equivalent) with Conv2d(512, num_classes, kernel_size=1).
    SqueezeNet 1.1 classifier = [Dropout, Conv2d(512, 1000, 1), ReLU, AdaptiveAvgPool2d((1,1))]
    """
    weights = SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = squeezenet1_1(weights=weights)

    # Replace conv10
    in_channels = model.classifier[1].in_channels  # 512
    model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    # Keep ReLU and AdaptiveAvgPool2d as-is
    return model


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler=None, criterion=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
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
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits.detach(), targets) * bs
        n += bs

    return running_loss / n, running_acc / n


@torch.no_grad()
def validate(model, loader, device, criterion=None):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, targets) * bs
        n += bs

    return running_loss / n, running_acc / n


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SqueezeNet on 4-class brain tumor dataset")
    parser.add_argument("--train_dir", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/data")
    parser.add_argument("--train_csv", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/image_labels.csv")
    parser.add_argument("--val_dir",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/data")
    parser.add_argument("--val_csv",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/image_labels.csv")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default= 5e-5, help="Base LR for transferred layers")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="squeezenet_best.pth")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets / loaders
    train_tfms = build_transforms(args.img_size, train=True)
    val_tfms   = build_transforms(args.img_size, train=False)

    train_ds = MRIBrainTumorCSV(args.train_dir, args.train_csv, transform=train_tfms)
    val_ds   = MRIBrainTumorCSV(args.val_dir, args.val_csv, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, drop_last=False)

    # Model
    model = build_squeezenet(num_classes=4, pretrained=not args.no_pretrained).to(device)

    # ======= LR factors for the new last conv layer ("conv10" eqv.) =======
    # 10x LR for new layer weight and bias vs base LR for transferred layers
    new_conv = model.classifier[1]  # nn.Conv2d(512, 4, kernel_size=1)
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p is new_conv.weight or (new_conv.bias is not None and p is new_conv.bias):
            continue
        base_params.append(p)

    param_groups = [
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": [new_conv.weight], "lr": args.lr * 10.0, "weight_decay": args.weight_decay},  # WeightLearnRateFactor=10
    ]
    if new_conv.bias is not None:
        param_groups.append({"params": [new_conv.bias], "lr": args.lr * 10.0, "weight_decay": 0.0})  # BiasLearnRateFactor=10

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Train loop
    best_acc, best_path = 0.0, None
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save_dir) / args.save_name

    print(f"Classes: {CLASS_NAMES}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"Using device: {device}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scaler, criterion)
        va_loss, va_acc = validate(model, val_loader, device, criterion)
        scheduler.step()

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
              f"{dt:.1f}s")

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
