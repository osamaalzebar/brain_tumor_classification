# ensemble_prob_head.py
import os
import random
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import timm

# -----------------------
# Paths / Config
# -----------------------
TRAIN_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/data"
TRAIN_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/train/Image_labels.csv"

VAL_IMG_ROOT   = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/data"
VAL_CSV        = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/val/Image_labels.csv"

OUTPUT_DIR     = "./outputs_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BEST_HEAD_PATH = os.path.join(OUTPUT_DIR, "ensemble_head_best.pth")

NUM_CLASSES = 4
EPOCHS = 30
BATCH_SIZE = 4
LR = 5e-5
NUM_WORKERS = 4
SEED = 42

# -----------------------
# Checkpoints (EDIT THESE)
# -----------------------
RESNET50_CKPT    = "./outputs_resnet50/resnet50_best.pth"
DENSENET201_CKPT = "./outputs_densenet201/densenet201_best.pth"
MOBILENETV2_CKPT = "./outputs_mobilenetv2/mobilenetv2_best.pth"
INCEPT_V3_CKPT   = "./outputs_inceptionv3/inceptionv3_best.pth"
XCEPTION_CKPT    = "./outputs_xception/xception_best.pth"

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
# Transforms (shared policy, two sizes)
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k)

def build_transforms(img_size: int, train: bool) -> Callable:
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

TFM_224_TRAIN = build_transforms(224, True)
TFM_224_VAL   = build_transforms(224, False)
TFM_299_TRAIN = build_transforms(299, True)
TFM_299_VAL   = build_transforms(299, False)

# -----------------------
# Dataset (returns PIL + label; labels 1..4 in CSV -> 0..3)
# -----------------------
class CsvImageDataset(Dataset):
    def __init__(self, img_root: str, csv_path: str):
        self.img_root = img_root
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if "Image_path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have header 'Image_path,label'")
        samples: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            img_name = str(row["Image_path"]).strip()
            label_1_to_4 = int(row["label"])
            path = os.path.join(self.img_root, os.path.basename(img_name))
            samples.append((path, label_1_to_4 - 1))
        self.samples = [(p, y) for (p, y) in samples if os.path.isfile(p)]
        missing = len(samples) - len(self.samples)
        if missing > 0:
            print(f"Warning: {missing} images in CSV not found under {self.img_root}; skipped.")
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples for {csv_path} with root {img_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, y

# -----------------------
# Backbone builders (must match your training scripts)
# -----------------------
def build_resnet50():
    m = models.resnet50(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, NUM_CLASSES)
    return m

def build_densenet201():
    m = models.densenet201(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Linear(in_f, NUM_CLASSES)
    return m

def build_mobilenetv2():
    m = models.mobilenet_v2(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
    return m

def build_inception_v3():
    # aux_logits True is required for torchvision pretrained; for inference we only use main logits
    m = models.inception_v3(weights=None, aux_logits=True)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, NUM_CLASSES)
    if m.aux_logits and m.AuxLogits is not None:
        aux_in = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(aux_in, NUM_CLASSES)
    return m

def build_xception():
    m = timm.create_model("xception", pretrained=False, num_classes=NUM_CLASSES)
    return m

def load_ckpt(model: nn.Module, path: str, key: str = "model_state_dict"):
    sd = torch.load(path, map_location="cpu")
    state = sd[key] if isinstance(sd, dict) and key in sd else sd
    model.load_state_dict(state, strict=True)
    return model

@torch.no_grad()
def inception_main_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple) and len(output) >= 1:
        return output[0]
    return output  # Tensor

# -----------------------
# Ensemble wrapper: frozen backbones, trainable head
# -----------------------
class ProbEnsembler(nn.Module):
    def __init__(self, resnet, densenet, mobilenet, inception, xception):
        super().__init__()
        # register submodules
        self.resnet = resnet.eval()
        self.densenet = densenet.eval()
        self.mobilenet = mobilenet.eval()
        self.inception = inception.eval()
        self.xception = xception.eval()

        # freeze
        for p in self.parameters():
            p.requires_grad = False

        # small head: 4 -> 64 -> Drop(0.5) -> 4 (logits)
        self.head = nn.Sequential(
            nn.Linear(NUM_CLASSES, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, NUM_CLASSES)
        )
        # unfreeze only head
        for p in self.head.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    def forward(self,
                img_224: torch.Tensor,
                img_299: torch.Tensor) -> torch.Tensor:
        """
        img_224: tensor batch [B,3,224,224]  -> for resnet/densenet/mobilenet
        img_299: tensor batch [B,3,299,299]  -> for inception/xception
        Returns: head logits [B,4]
        """
        # No grad through backbones
        with torch.no_grad():
            p1 = self._softmax(self.resnet(img_224))                      # resnet50
            p2 = self._softmax(self.densenet(img_224))                    # densenet201
            p3 = self._softmax(self.mobilenet(img_224))                   # mobilenetv2

            inc_out = self.inception(img_299)
            p4 = self._softmax(inception_main_logits(inc_out))            # inceptionv3

            p5 = self._softmax(self.xception(img_299))                    # xception (timm)

            p_avg = (p1 + p2 + p3 + p4 + p5) / 5.0                        # [B,4], probabilities

        # Trainable head (returns logits; sigmoid applied only for inference if needed)
        head_logits = self.head(p_avg)
        return head_logits

# -----------------------
# Utils
# -----------------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def collate_pil(batch):
    # batch: List[(PIL.Image, int)]
    imgs, ys = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long)

# -----------------------
# Train / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()   # head trainable, backbones frozen anyway
    else:
        model.eval()

    total_loss = total_acc = total_n = 0

    for pil_list, labels in loader:
        labels = labels.to(device, non_blocking=True)

        # Build two views with identical augmentation policy but different sizes
        imgs_224 = torch.stack([TFM_224_TRAIN(img) for img in pil_list]).to(device, non_blocking=True) if train \
                   else torch.stack([TFM_224_VAL(img) for img in pil_list]).to(device, non_blocking=True)
        imgs_299 = torch.stack([TFM_299_TRAIN(img) for img in pil_list]).to(device, non_blocking=True) if train \
                   else torch.stack([TFM_299_VAL(img) for img in pil_list]).to(device, non_blocking=True)

        logits = model(imgs_224, imgs_299)  # [B,4] head logits

        # BCEWithLogitsLoss expects multi-hot targets
        targets_oh = one_hot(labels, NUM_CLASSES).to(device)
        loss = criterion(logits, targets_oh)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(logits, labels) * bs
        total_n    += bs

    return total_loss / total_n, total_acc / total_n

# -----------------------
# Main
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets / Loaders (dataset returns PIL so we can apply both sizes)
    train_ds = CsvImageDataset(TRAIN_IMG_ROOT, TRAIN_CSV)
    val_ds   = CsvImageDataset(VAL_IMG_ROOT, VAL_CSV)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_pil)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=collate_pil)

    # Build backbones and load YOUR fine-tuned weights
    resnet    = load_ckpt(build_resnet50(),    RESNET50_CKPT)
    densenet  = load_ckpt(build_densenet201(), DENSENET201_CKPT)
    mobilenet = load_ckpt(build_mobilenetv2(), MOBILENETV2_CKPT)
    inception = load_ckpt(build_inception_v3(), INCEPT_V3_CKPT)
    xception  = load_ckpt(build_xception(),    XCEPTION_CKPT)

    # Ensemble model (frozen backbones, trainable head)
    model = ProbEnsembler(resnet, densenet, mobilenet, inception, xception).to(device)

    # Only head params trainable
    params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc     = train_one_epoch(model, val_loader,   optimizer, criterion, device, train=False)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%   "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "head_state_dict": model.head.state_dict(),
                "val_acc": best_val_acc,
                "config": {
                    "num_classes": NUM_CLASSES,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR
                }
            }, BEST_HEAD_PATH)
            print(f"  ↳ Saved best head (Val Acc: {best_val_acc*100:.2f}%) to {BEST_HEAD_PATH}")

    print("Training complete. Best Val Acc: {:.2f}%".format(best_val_acc*100))

    # Optional: show example final probabilities on one val batch
    model.eval()
    with torch.no_grad():
        for pil_list, labels in val_loader:
            imgs_224 = torch.stack([TFM_224_VAL(img) for img in pil_list]).to(device)
            imgs_299 = torch.stack([TFM_299_VAL(img) for img in pil_list]).to(device)
            logits = model(imgs_224, imgs_299)
            probs = torch.sigmoid(logits)  # per your request (sigmoid at output)
            print("Example ensembled head probabilities (first batch):")
            print(probs[:min(4, probs.size(0))].cpu())
            break

if __name__ == "__main__":
    main()
