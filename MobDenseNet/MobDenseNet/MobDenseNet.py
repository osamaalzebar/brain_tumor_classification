# train_mobilenetv1_densenet121_concat.py
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
# Config (same paths/hparams as earlier)
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

OUTPUT_DIR     = "./outputs_mobv1_densenet121_concat"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "mobv1_densenet121_concat_best.pth")

NUM_CLASSES = 4
IMG_SIZE = 224         # same augmentation size as earlier scripts; change to 256 if you want to match the diagram
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
# Transforms (same policy you used)
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
# Dual-backbone concatenation model
#   MobileNetV1 (A) + DenseNet121 (B)
#   -> AdaptiveMaxPool2d(1) -> Flatten
#   -> Concatenate -> BN -> Dense(1280, ReLU, GlorotUniform seed=1377, bias=0)
#   -> BN -> Classifier(4 logits)
# -----------------------
class MobV1Dense121Concat(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # Use timm for both so we can easily grab feature maps
        self.backbone_a = timm.create_model("mobilenetv1_100", pretrained=True, num_classes=0, global_pool="")
        self.backbone_b = timm.create_model("densenet121",     pretrained=True, num_classes=0, global_pool="")

        self.pool = nn.AdaptiveMaxPool2d(1)  # "Max Pool" in your diagram

        self.feat_dim_a = getattr(self.backbone_a, "num_features")
        self.feat_dim_b = getattr(self.backbone_b, "num_features")
        concat_dim = self.feat_dim_a + self.feat_dim_b

        self.bn_concat = nn.BatchNorm1d(concat_dim)

        self.fc1 = nn.Linear(concat_dim, 1280)
        self.relu = nn.ReLU(inplace=True)
        self.bn_fc1 = nn.BatchNorm1d(1280)

        self.classifier = nn.Linear(1280, num_classes)  # logits for CE

        # Init fc1 with Glorot/Xavier-Uniform (seed=1377), bias zeros
        self._init_fc1_glorot(seed=1377)

    def _init_fc1_glorot(self, seed: int = 1377):
        try:
            gen = torch.Generator(device=self.fc1.weight.device)
            gen.manual_seed(seed)
            nn.init.xavier_uniform_(self.fc1.weight, gain=1.0, generator=gen)
        except TypeError:
            # Fallback for older PyTorch without 'generator' arg
            cur = torch.random.get_rng_state()
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
            torch.random.set_rng_state(cur)
        nn.init.zeros_(self.fc1.bias)

    def extract_flatten(self, backbone, x):
        feat = backbone.forward_features(x)   # [B, C, H, W]
        feat = self.pool(feat)                # [B, C, 1, 1]
        return feat.flatten(1)                # [B, C]

    def forward(self, x, return_logits: bool = True):
        fA = self.extract_flatten(self.backbone_a, x)  # [B, Ca]
        fB = self.extract_flatten(self.backbone_b, x)  # [B, Cb]
        z  = torch.cat([fA, fB], dim=1)                # [B, Ca+Cb]

        z  = self.bn_concat(z)
        z  = self.fc1(z)
        z  = self.relu(z)
        z  = self.bn_fc1(z)

        logits = self.classifier(z)                    # [B, 4]
        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)

# -----------------------
# Metrics / loops
# -----------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = running_acc = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs, return_logits=True)
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

        logits = model(imgs, return_logits=True)
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

    model = MobV1Dense121Concat(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
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

    # Optional: quick probability print
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            probs = model(imgs, return_logits=False)
            print("Example softmax probabilities (first batch):")
            print(probs[:min(4, probs.size(0))].cpu())
            break

if __name__ == "__main__":
    main()
