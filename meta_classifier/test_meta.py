# test_ensemble_prob_head.py
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import timm

# -----------------------
# Paths / Config (EDIT THESE)
# -----------------------
TEST_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/data"
TEST_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/Image_labels.csv"

# Checkpoints (should match your training outputs)
RESNET50_CKPT    = "./outputs_resnet50/resnet50_best.pth"
DENSENET201_CKPT = "./outputs_densenet201/densenet201_best.pth"
MOBILENETV2_CKPT = "./outputs_mobilenetv2/mobilenetv2_best.pth"
INCEPT_V3_CKPT   = "./outputs_inceptionv3/inceptionv3_best.pth"
XCEPTION_CKPT    = "./outputs_xception/xception_best.pth"
HEAD_CKPT        = "./outputs_ensemble/ensemble_head_best.pth"

NUM_CLASSES = 4
BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]  # optional, for nicer prints

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
# Transforms (eval only)
# -----------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TFM_224_VAL = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

TFM_299_VAL = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------
# Dataset (CSV: Image_path,label; labels 1..4 -> 0..3)
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

def collate_pil(batch):
    imgs, ys = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long)

# -----------------------
# Backbone builders (match training definitions)
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
    m = models.inception_v3(weights=None, aux_logits=True)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, NUM_CLASSES)
    if m.aux_logits and m.AuxLogits is not None:
        aux_in = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(aux_in, NUM_CLASSES)
    return m

def build_xception():
    return timm.create_model("xception", pretrained=False, num_classes=NUM_CLASSES)

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
    return output

# -----------------------
# Ensemble wrapper (same as training; backbones frozen)
# -----------------------
class ProbEnsembler(nn.Module):
    def __init__(self, resnet, densenet, mobilenet, inception, xception):
        super().__init__()
        self.resnet = resnet.eval()
        self.densenet = densenet.eval()
        self.mobilenet = mobilenet.eval()
        self.inception = inception.eval()
        self.xception = xception.eval()

        # freeze all backbones
        for p in self.resnet.parameters(): p.requires_grad = False
        for p in self.densenet.parameters(): p.requires_grad = False
        for p in self.mobilenet.parameters(): p.requires_grad = False
        for p in self.inception.parameters(): p.requires_grad = False
        for p in self.xception.parameters(): p.requires_grad = False

        # head must be defined and loaded separately
        self.head = nn.Sequential(
            nn.Linear(NUM_CLASSES, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, NUM_CLASSES)
        )

    @torch.no_grad()
    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=1)

    def forward(self, img_224: torch.Tensor, img_299: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            p1 = self._softmax(self.resnet(img_224))
            p2 = self._softmax(self.densenet(img_224))
            p3 = self._softmax(self.mobilenet(img_224))
            inc_out = self.inception(img_299)
            p4 = self._softmax(inception_main_logits(inc_out))
            p5 = self._softmax(self.xception(img_299))
            p_avg = (p1 + p2 + p3 + p4 + p5) / 5.0  # [B,4]
        logits = self.head(p_avg)  # [B,4] (trained with BCEWithLogitsLoss)
        return logits

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

@torch.no_grad()
def per_class_accuracy(cm: np.ndarray):
    # diag / row sum
    accs = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        accs.append((cm[i, i] / denom) if denom > 0 else 0.0)
    return accs

# -----------------------
# Main (test)
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Data
    test_ds = CsvImageDataset(TEST_IMG_ROOT, TEST_CSV)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             collate_fn=collate_pil)

    # Build backbones and load weights
    resnet    = load_ckpt(build_resnet50(),    RESNET50_CKPT)
    densenet  = load_ckpt(build_densenet201(), DENSENET201_CKPT)
    mobilenet = load_ckpt(build_mobilenetv2(), MOBILENETV2_CKPT)
    inception = load_ckpt(build_inception_v3(), INCEPT_V3_CKPT)
    xception  = load_ckpt(build_xception(),    XCEPTION_CKPT)

    # Assemble ensemble and load head
    model = ProbEnsembler(resnet, densenet, mobilenet, inception, xception).to(device)
    # Load best head weights
    head_sd = torch.load(HEAD_CKPT, map_location="cpu")
    model.head.load_state_dict(head_sd["head_state_dict"], strict=True)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pil_list, labels in test_loader:
            labels = labels.to(device, non_blocking=True)
            imgs_224 = torch.stack([TFM_224_VAL(img) for img in pil_list]).to(device, non_blocking=True)
            imgs_299 = torch.stack([TFM_299_VAL(img) for img in pil_list]).to(device, non_blocking=True)

            logits = model(imgs_224, imgs_299)  # [B,4]
            # You trained head with BCEWithLogitsLoss; argmax over logits == argmax over sigmoid(logits)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Metrics
    overall_acc = (y_pred == y_true).mean()
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    cls_acc = per_class_accuracy(cm)
    macro_acc = float(np.mean(cls_acc))

    print("\n=== Ensemble Test Results ===")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print("Per-class Accuracy:")
    for i, a in enumerate(cls_acc):
        cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i} ({cname}): {a*100:.2f}%")
    print(f"Macro-average Accuracy: {macro_acc*100:.2f}%")

    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(cm)

    # Optionally save confusion matrix to CSV
    out_dir = os.path.dirname(HEAD_CKPT) if os.path.dirname(HEAD_CKPT) else "."
    cm_path = os.path.join(out_dir, "confusion_matrix_test.csv")
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(NUM_CLASSES)],
                    columns=[f"pred_{i}" for i in range(NUM_CLASSES)]).to_csv(cm_path)
    print(f"\nSaved confusion matrix to: {cm_path}")

if __name__ == "__main__":
    main()
