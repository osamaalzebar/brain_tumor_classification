# fine_tune_resnet50v2_custom_head_v2.py
import os
import random
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm

# -----------------------
# Config (edit paths if needed)
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

OUTPUT_DIR     = "./outputs_resnet50v2_custom"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "resnet50v2_custom_best.pth")

NUM_CLASSES = 4
IMG_SIZE = 224
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
# Transforms (your policy)
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
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
# Dataset (CSV: Image_path,label ; labels 1..4 -> 0..3)
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
            img_name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            candidate = os.path.join(self.img_root, os.path.basename(img_name))
            samples.append((candidate, label_1_to_4 - 1))

        existing = [(p, y) for (p, y) in samples if os.path.isfile(p)]
        missing = len(samples) - len(existing)
        if missing > 0:
            print(f"Warning: {missing} images listed in CSV not found under {self.img_root}. Skipping.")
        self.samples = existing
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples for csv={csv_path} with root={img_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, target

# -----------------------
# Model: ResNet50V2 backbone (pretrained) + custom head exactly as requested
#   GAP -> BN -> Dense(1280, ReLU, GlorotUniform seed=1377, bias=0)
#       -> BN -> Dense(4) -> Softmax (for inference)
# Training uses logits pre-softmax for CE loss (stable).
# -----------------------
class ResNet50V2Custom(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Pre-activation ResNet-50 (ResNetV2). We truncate by using features only.
        self.backbone = timm.create_model("resnetv2_50", pretrained=True, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features  # typically 2048

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm1d(feat_dim)

        self.fc1 = nn.Linear(feat_dim, 1280)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(1280)

        self.classifier = nn.Linear(1280, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # --- Initializers for fc1 (Glorot/Xavier-Uniform with seed=1377), bias zeros
        self._init_fc1_glorot(seed=1377)

    def _init_fc1_glorot(self, seed: int = 1377):
        try:
            gen = torch.Generator(device=self.fc1.weight.device)
            gen.manual_seed(seed)
            nn.init.xavier_uniform_(self.fc1.weight, gain=1.0, generator=gen)
        except TypeError:
            # Fallback if PyTorch doesn't support 'generator' in this version
            current = torch.random.get_rng_state()
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
            torch.random.set_rng_state(current)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x, return_logits: bool = False):
        feats = self.backbone.forward_features(x)      # [B, C, H, W]
        x = self.gap(feats).flatten(1)                 # [B, C]
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        logits = self.classifier(x)                    # [B, num_classes]
        if return_logits:
            return logits
        return self.softmax(logits)                    # final Softmax for inference

# -----------------------
# Metrics / loops
# -----------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = running_acc = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, return_logits=True)  # use logits for CE
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = running_acc = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs, return_logits=True)  # logits for CE
        loss = criterion(logits, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy_from_logits(logits, labels) * bs
        total += bs

    return running_loss / total, running_acc / total

# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms = build_transforms(IMG_SIZE, train=True)
    val_tfms   = build_transforms(IMG_SIZE, train=False)

    train_ds = CsvImageDataset(TRAIN_IMG_ROOT, TRAIN_CSV, transform=train_tfms)
    val_ds   = CsvImageDataset(VAL_IMG_ROOT, VAL_CSV,   transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = ResNet50V2Custom(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()  # logits + CE (softmax is applied only for inference)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%   "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save best by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "img_size": IMG_SIZE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR
                }
            }, BEST_MODEL_PATH)
            print(f"  ↳ Saved best model (Val Acc: {best_val_acc*100:.2f}%) to {BEST_MODEL_PATH}")

    print("Training complete. Best Val Acc: {:.2f}%".format(best_val_acc*100))

    # Optional: example softmax probabilities from the model (inference mode)
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            probs = model(imgs, return_logits=False)   # softmax probs
            print("Example softmax probabilities (first batch):")
            print(probs[:min(4, probs.size(0))].cpu())
            break

if __name__ == "__main__":
    main()
