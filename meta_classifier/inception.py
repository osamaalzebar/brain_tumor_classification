# fine_tune_inceptionv3_brain_tumor.py
import os
import random
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# -----------------------
# Config (edit if needed)
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"  # adjust if different
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

OUTPUT_DIR     = "./outputs_inceptionv3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "inceptionv3_best.pth")

NUM_CLASSES = 4
IMG_SIZE = 299  # InceptionV3 input size
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
# Transforms
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k)

def build_transforms(img_size: int = 299, train: bool = True) -> Callable:
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
            img_name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            candidate = os.path.join(self.img_root, os.path.basename(img_name))
            samples.append((candidate, label_1_to_4 - 1))  # shift to 0..3

        existing = [(p, y) for (p, y) in samples if os.path.isfile(p)]
        missing = len(samples) - len(existing)
        if missing > 0:
            print(f"Warning: {missing} images listed in CSV not found under {self.img_root}. Skipping.")
        self.samples = existing
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples for csv={csv_path} with root={img_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# -----------------------
# Model: InceptionV3 with aux logits ON (required with pretrained weights)
# -----------------------
def build_inception_v3(num_classes: int = 4) -> nn.Module:
    # When using pretrained weights, torchvision enforces aux_logits=True
    model = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        aux_logits=True
    )
    # Replace main classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Replace aux classifier if present
    if model.aux_logits and model.AuxLogits is not None:
        aux_in = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
    return model

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total

def _split_inception_outputs(output):
    """
    Robustly extract (main_logits, aux_logits_or_None) from InceptionV3 forward().
    - In train mode with aux: may return a namedtuple with .logits and .aux_logits
    - In some versions: may be a tuple (main, aux)
    - In eval: usually a plain Tensor (main logits)
    """
    # Named tuple with attributes
    if hasattr(output, "logits"):
        return output.logits, getattr(output, "aux_logits", None)
    # Tuple (main, aux) case
    if isinstance(output, tuple) and len(output) == 2:
        return output[0], output[1]
    # Plain tensor
    return output, None

# -----------------------
# Train / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_acc = total_count = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(imgs)
        main_logits, aux_logits = _split_inception_outputs(output)

        loss = criterion(main_logits, labels)
        if aux_logits is not None:
            loss = loss + 0.4 * criterion(aux_logits, labels)  # standard aux loss

        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(main_logits, labels) * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = total_count = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output = model(imgs)
        main_logits, _ = _split_inception_outputs(output)  # ignore aux at eval
        loss = criterion(main_logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(main_logits, labels) * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count

# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms = build_transforms(IMG_SIZE, train=True)
    val_tfms   = build_transforms(IMG_SIZE, train=False)

    train_ds = CsvImageDataset(TRAIN_IMG_ROOT, TRAIN_CSV, transform=train_tfms)
    val_ds   = CsvImageDataset(VAL_IMG_ROOT, VAL_CSV, transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = build_inception_v3(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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

    print("Training complete.")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")

    # Optional: example softmax probabilities on one val batch
    model.eval()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits, _ = _split_inception_outputs(model(imgs))
            probs = torch.softmax(logits, dim=1)
            print("Example softmax probabilities (first batch):")
            print(probs[:min(4, probs.size(0))].cpu())
            break

if __name__ == "__main__":
    main()
