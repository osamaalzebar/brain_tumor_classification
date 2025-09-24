# train_frozen_feature_heads_7models.py
import os
import random
from typing import Callable, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm

# -----------------------
# Paths / Hyperparams (same as before)
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

ROOT_OUT = "./outputs_frozen_heads_7models"
os.makedirs(ROOT_OUT, exist_ok=True)

NUM_CLASSES = 4
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -----------------------
# Dataset (CSV: Image_path,label; labels 1..4 -> 0..3)
# -----------------------
class CsvImageDataset(Dataset):
    def __init__(self, img_root: str, csv_path: str, transform: Callable):
        self.img_root = img_root
        self.transform = transform

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have columns: 'Image_path,label'")

        samples: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            path = os.path.join(self.img_root, os.path.basename(name))
            samples.append((path, label_1_to_4 - 1))

        self.samples = [(p, y) for (p, y) in samples if os.path.isfile(p)]
        miss = len(samples) - len(self.samples)
        if miss > 0:
            print(f"Warning: {miss} images listed in CSV not found under {self.img_root}. Skipping.")
        if not self.samples:
            raise RuntimeError(f"No valid samples for csv={csv_path} with root={img_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y

# -----------------------
# Augmentation (RandomRotate90 + per-model size/mean/std)
# -----------------------
class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k)

def build_transforms(img_size: int, mean, std, train: bool) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

# -----------------------
# Frozen backbone + three Swish dense layers head
# -----------------------
class FrozenFeatureHead(nn.Module):
    """
    backbone: timm model with num_classes=0, global_pool=''
    head: Flatten -> 3 x (Linear 1024 + Swish) -> Dropout(0.2) -> Linear NUM_CLASSES
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 4):
        super().__init__()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        feat_dim = getattr(self.backbone, "num_features")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

        self.fc1 = nn.Linear(feat_dim, 1024)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.act3 = nn.SiLU()

        self.drop = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, num_classes)

        # Xavier init for dense layers (incl. classifier); bias zeros
        for layer in [self.fc1, self.fc2, self.fc3, self.classifier]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, return_logits: bool = True):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)  # [B, C, H, W]
        x = self.gap(feats)
        x = self.flatten(x)

        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.drop(x)
        logits = self.classifier(x)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)

# -----------------------
# Train / Eval loops (shared)
# -----------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # only head params require grad
    tot_loss = tot_acc = tot_n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, return_logits=True)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_acc  += accuracy_from_logits(logits, labels) * bs
        tot_n    += bs
    return tot_loss / tot_n, tot_acc / tot_n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_acc = tot_n = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs, return_logits=True)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        tot_loss += loss.item() * bs
        tot_acc  += accuracy_from_logits(logits, labels) * bs
        tot_n    += bs
    return tot_loss / tot_n, tot_acc / tot_n

# -----------------------
# Utilities to read timm config (size / mean / std)
# -----------------------
def get_preproc_from_timm(model: nn.Module):
    # timm >= 0.9: model.pretrained_cfg; older: default_cfg
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std  = cfg.get("std",  (0.229, 0.224, 0.225))
    size = cfg.get("input_size", (3, 224, 224))
    img_size = size[-1] if isinstance(size, (tuple, list)) else int(size)
    return img_size, list(mean), list(std)

# -----------------------
# Per-model definitions (timm names)
# -----------------------
MODELS = [
    ("vgg19",                 "VGG19"),
    ("resnetv2_50",           "ResNet50V2"),
    ("inception_v3",          "InceptionV3"),
    ("inception_resnet_v2",   "InceptionResNetV2"),
    ("densenet201",           "DenseNet201"),
    ("mobilenetv2_100",       "MobileNetV2"),
    ("tf_efficientnet_b7",    "EfficientNetB7"),
]

# -----------------------
# Train one model end-to-end
# -----------------------
def train_model_timm(model_name: str, pretty: str, device: torch.device):
    print(f"\n====== Training {pretty} (frozen backbone) ======")

    # Backbone (pretrained, feature-only)
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="")
    img_size, mean, std = get_preproc_from_timm(backbone)

    out_dir = os.path.join(ROOT_OUT, f"{pretty.lower().replace(' ', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{pretty.lower().replace(' ', '_')}_best.pth")

    # Data
    train_tfms = build_transforms(img_size, mean, std, train=True)
    val_tfms   = build_transforms(img_size, mean, std, train=False)

    train_ds = CsvImageDataset(TRAIN_IMG_ROOT, TRAIN_CSV, transform=train_tfms)
    val_ds   = CsvImageDataset(VAL_IMG_ROOT,   VAL_CSV,   transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Model (frozen backbone + head)
    model = FrozenFeatureHead(backbone, num_classes=NUM_CLASSES).to(device)

    # Optimizer only on trainable params (the head)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(f"[{pretty}] Epoch {epoch:02d}/{EPOCHS}  "
              f"Train Loss {tr_loss:.4f}  Train Acc {tr_acc*100:.2f}%   "
              f"Val Loss {va_loss:.4f}  Val Acc {va_acc*100:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "preproc": {"img_size": img_size, "mean": mean, "std": std},
                "model_name": model_name,
                "pretty_name": pretty,
                "config": {
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "num_classes": NUM_CLASSES,
                }
            }, best_path)
            print(f"   ↳ Saved best model to {best_path} (Val Acc {best_val_acc*100:.2f}%)")

    print(f"[{pretty}] Training complete. Best Val Acc: {best_val_acc*100:.2f}%")

# -----------------------
# Main: loop over the 7 models
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    for timm_name, pretty in MODELS:
        train_model_timm(timm_name, pretty, device)

if __name__ == "__main__":
    main()
