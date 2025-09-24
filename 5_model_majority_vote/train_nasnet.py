import argparse
import os
from datetime import datetime
from typing import Tuple, List
import csv
import random
import ssl
from urllib.error import URLError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---- deps: pretrainedmodels (NASNet-Mobile lives here) ----
import pretrainedmodels

# ---------------------------
# Dataset + Transforms
# ---------------------------

class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k, expand=True)

def build_transforms(
    img_size: int,
    train: bool,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

class BrainTumorCSVDataset(Dataset):
    """
    CSV header: Image_path,label
      - Image_path is filename inside images_dir
      - label in {1,2,3,4} -> mapped to {0,1,2,3}
    """
    def __init__(self, images_dir: str, labels_csv: str, transform=None):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self._load()

    def _load(self):
        if not os.path.isdir(self.images_dir):
            raise NotADirectoryError(self.images_dir)
        if not os.path.isfile(self.labels_csv):
            raise FileNotFoundError(self.labels_csv)

        with open(self.labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames, \
                "CSV must have header: Image_path,label"

            for row in reader:
                fn = row["Image_path"].strip()
                label_raw = int(row["label"])
                label = label_raw - 1  # map 1..4 -> 0..3
                path = os.path.join(self.images_dir, fn)
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                self.samples.append((path, label))

        if len(self.samples) == 0:
            raise ValueError("No samples found")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # start grayscale
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ---------------------------
# Model: NASNet-Mobile
# ---------------------------

class NASNetMobileClassifier(nn.Module):
    """
    Wraps NASNet-Mobile with a new 4-class head (Linear), plus a Softmax layer
    for probabilities when requested.

    - "Fully connected layer"  -> nn.Linear(in_features, 4)
    - "Softmax layer"          -> nn.Softmax(dim=1) (used when return_probs=True)
    - "Classification output"  -> argmax of probabilities (for evaluation/inference)
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 4):
        super().__init__()
        self.backbone = backbone  # backbone.last_linear already replaced
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_probs: bool = False):
        logits = self.backbone(x)        # logits shape [B, 4]
        if return_probs:
            probs = self.softmax(logits) # probabilities
            preds = torch.argmax(probs, dim=1)
            return logits, probs, preds
        return logits

def load_nasnet_mobile_pretrained_backbone(num_classes: int = 4):
    """
    Builds NASNet-Mobile from pretrainedmodels, loads ImageNet weights for the BACKBONE,
    drops the 1000-class head, and attaches a fresh 4-class head.
    Returns: model (wrapped), mean, std, img_size
    """
    # 1) Instantiate NASNet-Mobile with NO pretrained head
    base = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained=None)

    # 2) Get official preprocessing (mean/std, input size, URL)
    setting = pretrainedmodels.pretrained_settings['nasnetamobile']['imagenet']
    mean = tuple(setting['mean'])
    std = tuple(setting['std'])
    img_size = setting.get('input_size', (3, 224, 224))[1]
    url = setting['url']

    # 3) Download pretrained weights with TLS verification disabled (site has expired cert)
    try:
        print("[INFO] Downloading NASNet-Mobile pretrained backbone (TLS verify disabled once).")
        orig_ctx = getattr(ssl, "_create_default_https_context", None)
        ssl._create_default_https_context = ssl._create_unverified_context
        state = torch.hub.load_state_dict_from_url(url, progress=True, model_dir=os.path.expanduser("~/.cache/torch/hub/checkpoints"))
        if orig_ctx is not None:
            ssl._create_default_https_context = orig_ctx
    except (URLError, ssl.SSLError, Exception) as e:
        print(f"[WARN] Could not download pretrained weights: {e}")
        state = None

    # 4) Replace head with a fresh 4-class Linear (our new "fully connected layer")
    in_features = base.last_linear.in_features
    base.last_linear = nn.Linear(in_features, num_classes)

    # 5) Load ONLY the backbone weights (drop the 1000-class head keys)
    if state is not None:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        removed = []
        for k in ["last_linear.weight", "last_linear.bias"]:
            if k in state:
                state.pop(k)
                removed.append(k)
        missing, unexpected = base.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded pretrained backbone. Removed head keys: {removed}")
        if missing:
            print(f"[INFO] Missing (first few): {missing[:6]}{'...' if len(missing)>6 else ''}")
        if unexpected:
            print(f"[INFO] Unexpected (first few): {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")
    else:
        print("[WARN] Proceeding with RANDOMLY INITIALIZED weights (will still fine-tune).")

    # 6) Wrap with classifier exposing softmax & class outputs on demand
    model = NASNetMobileClassifier(base, num_classes=num_classes)
    return model, mean, std, img_size

# ---------------------------
# Training utilities
# ---------------------------

CLASSES = ["meningioma", "glioma", "pituitary", "no_tumor"]

def top1_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        return 100.0 * (preds == targets).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += top1_acc(logits, targets) * bs
        total += bs

    return running_loss / total, running_acc / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, targets)

        bs = targets.size(0)
        running_loss += loss.item() * bs
        running_acc += top1_acc(logits, targets) * bs
        total += bs

    return running_loss / total, running_acc / total

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser("Fine-tune NASNet-Mobile for brain tumor MRI (4 classes)")
    parser.add_argument("--train_images", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/data")
    parser.add_argument("--train_csv",    type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/train/image_labels.csv")
    parser.add_argument("--val_images",   type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/data")
    parser.add_argument("--val_csv",      type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/majority_paper/classification_task/val/image_labels.csv")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build NASNet-Mobile (pretrained backbone) + get correct mean/std/img_size
    model, mean, std, img_size = load_nasnet_mobile_pretrained_backbone(num_classes=4)
    model = model.to(device)

    # Datasets / Loaders
    train_tfms = build_transforms(img_size=img_size, train=True,  mean=mean, std=std)
    val_tfms   = build_transforms(img_size=img_size, train=False, mean=mean, std=std)

    train_ds = BrainTumorCSVDataset(args.train_images, args.train_csv, transform=train_tfms)
    val_ds   = BrainTumorCSVDataset(args.val_images,   args.val_csv,   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Loss/Opt
    criterion = nn.CrossEntropyLoss()  # expects logits (no softmax)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume (optional)
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"=> Resumed from {args.resume} (epoch {start_epoch}, best_acc {best_val_acc:.2f}%)")

    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))

    # Train
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
              f"Val: loss={va_loss:.4f} acc={va_acc:.2f}%")

        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "timestamp": datetime.now().isoformat(),
            "classes": CLASSES,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last_nasnet_mobile.pth"))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(ckpt, os.path.join(args.save_dir, "best_nasnet_mobile.pth"))
            print(f"✓ New best model saved (val acc {best_val_acc:.2f}%).")

    print(f"Training finished. Best val acc: {best_val_acc:.2f}%")
    print("Targets: 0=meningioma, 1=glioma, 2=pituitary, 3=no_tumor")
    print(f"Best checkpoint: {os.path.join(args.save_dir, 'best_nasnet_mobile.pth')}")


if __name__ == "__main__":
    main()
