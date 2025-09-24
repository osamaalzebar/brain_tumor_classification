# test_7models_majority_vote.py
import os
import random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import timm

# -----------------------
# Paths / Config (EDIT THESE)
# -----------------------
TEST_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/data"
TEST_CSV      = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/SAlexNet/classification_task/test/Image_labels.csv"

# Where you saved best checkpoints from the 7-model training script
ROOT_OUT = "./outputs_frozen_heads_7models"
CKPTS = {
    "VGG19":              os.path.join(ROOT_OUT, "vgg19",              "vgg19_best.pth"),
    "ResNet50V2":         os.path.join(ROOT_OUT, "resnet50v2",         "resnet50v2_best.pth"),
    "InceptionV3":        os.path.join(ROOT_OUT, "inceptionv3",        "inceptionv3_best.pth"),
    "InceptionResNetV2":  os.path.join(ROOT_OUT, "inceptionresnetv2",  "inceptionresnetv2_best.pth"),
    "DenseNet201":        os.path.join(ROOT_OUT, "densenet201",        "densenet201_best.pth"),
    "MobileNetV2":        os.path.join(ROOT_OUT, "mobilenetv2",        "mobilenetv2_best.pth"),
    "EfficientNetB7":     os.path.join(ROOT_OUT, "efficientnetb7",     "efficientnetb7_best.pth"),
}

NUM_CLASSES = 4
BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]  # optional, for printing

# Save detailed per-image outputs?
SAVE_PREDICTIONS_CSV = True
OUTPUT_DIR = ROOT_OUT
os.makedirs(OUTPUT_DIR, exist_ok=True)
PREDICTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, "majority_vote_test_predictions.csv")
CM_CSV_PATH          = os.path.join(OUTPUT_DIR, "majority_vote_confusion_matrix.csv")

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
# Dataset (no augmentation)
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
        miss = len(samples) - len(self.samples)
        if miss > 0:
            print(f"Warning: {miss} images in CSV not found under {self.img_root}; skipped.")
        if not self.samples:
            raise RuntimeError(f"No valid samples in {csv_path} with root {img_root}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, y, os.path.basename(path)

def collate_pil(batch):
    imgs, ys, names = zip(*batch)
    return list(imgs), torch.tensor(ys, dtype=torch.long), list(names)

# -----------------------
# Head & wrapper (must match training)
# -----------------------
class FrozenFeatureHead(nn.Module):
    """Flatten -> 3x(Linear 1024 + Swish) -> Dropout(0.2) -> Linear 4"""
    def __init__(self, backbone: nn.Module, num_classes: int = 4):
        super().__init__()
        self.backbone = backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        feat_dim = getattr(self.backbone, "num_features")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(feat_dim, 1024); self.act1 = nn.SiLU()
        self.fc2 = nn.Linear(1024, 1024);     self.act2 = nn.SiLU()
        self.fc3 = nn.Linear(1024, 1024);     self.act3 = nn.SiLU()
        self.drop = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self, x, return_logits: bool = True):
        with torch.no_grad():
            feats = self.backbone.forward_features(x)
        x = self.gap(feats); x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.drop(x)
        logits = self.classifier(x)
        if return_logits: return logits
        return torch.softmax(logits, dim=1)

def get_preproc_from_cfg(cfg):
    if cfg is None: return 224, [0.485,0.456,0.406], [0.229,0.224,0.225]
    mean = cfg.get("mean", (0.485,0.456,0.406)); std = cfg.get("std", (0.229,0.224,0.225))
    size = cfg.get("input_size", (3,224,224))
    img_size = size[-1] if isinstance(size, (list,tuple)) else int(size)
    return img_size, list(mean), list(std)

def build_eval_tfms(img_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", None)
    preproc = ckpt.get("preproc", None)

    if model_name is None:
        raise ValueError(f"'model_name' not found in checkpoint: {ckpt_path}")

    # Build backbone and wrapper
    backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="")
    model = FrozenFeatureHead(backbone, num_classes=NUM_CLASSES).to(device)

    # Load weights
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Preprocess (use saved; fallback to timm default)
    if preproc is None:
        cfg = getattr(backbone, "pretrained_cfg", None) or getattr(backbone, "default_cfg", None)
        img_size, mean, std = get_preproc_from_cfg(cfg)
    else:
        img_size = int(preproc.get("img_size", 224))
        mean = preproc.get("mean", [0.485,0.456,0.406])
        std  = preproc.get("std",  [0.229,0.224,0.225])

    tfm = build_eval_tfms(img_size, mean, std)
    return model, tfm, img_size

# -----------------------
# Voting utilities
# -----------------------
def majority_vote_with_tiebreak(votes_row: np.ndarray, avg_probs_row: np.ndarray) -> int:
    """
    votes_row: [C] vote counts for one sample
    avg_probs_row: [C] average probs across models for one sample
    """
    max_votes = votes_row.max()
    cand = np.where(votes_row == max_votes)[0]
    if len(cand) == 1:
        return int(cand[0])
    # Tie-breaker: choose class with highest average probability among tied classes
    best = cand[np.argmax(avg_probs_row[cand])]
    # Final deterministic fallback (rare exact ties)
    if np.sum(avg_probs_row[cand] == avg_probs_row[best]) > 1:
        best = int(np.min(cand))
    return int(best)

@torch.no_grad()
def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_accuracy(cm: np.ndarray):
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

    # Load models + their eval transforms
    models_info = []  # list of dicts: {"name", "model", "tfm"}
    for name, path in CKPTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")
        model, tfm, img_size = load_model_from_ckpt(path, device)
        models_info.append({"name": name, "model": model, "tfm": tfm, "img_size": img_size})
        print(f"Loaded {name}  (img_size={img_size})")

    # Dataset/Loader (returns PIL to apply per-model tfms)
    ds = CsvImageDataset(TEST_IMG_ROOT, TEST_CSV)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_pil)

    all_true, all_pred = [], []
    all_names = []
    # Optional: store average probabilities & votes per image
    per_image_avg_probs = []
    per_image_votes = []

    with torch.no_grad():
        for pil_list, labels, names in loader:
            B = len(pil_list)
            labels_np = labels.numpy()
            # Accumulators for this batch
            votes = np.zeros((B, NUM_CLASSES), dtype=int)
            probs_sum = np.zeros((B, NUM_CLASSES), dtype=np.float32)

            for info in models_info:
                tfm = info["tfm"]; model = info["model"]
                batch = torch.stack([tfm(img) for img in pil_list]).to(device, non_blocking=True)
                logits = model(batch, return_logits=True)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # [B, C]
                preds = probs.argmax(axis=1)

                # accumulate votes and probs
                for i in range(B):
                    votes[i, preds[i]] += 1
                probs_sum += probs

            avg_probs = probs_sum / float(len(models_info))  # [B, C]
            final_preds = np.zeros(B, dtype=int)
            for i in range(B):
                final_preds[i] = majority_vote_with_tiebreak(votes[i], avg_probs[i])

            all_true.append(labels_np)
            all_pred.append(final_preds)
            all_names.extend(names)
            per_image_avg_probs.append(avg_probs)
            per_image_votes.append(votes)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    avg_probs_all = np.concatenate(per_image_avg_probs, axis=0)
    votes_all = np.concatenate(per_image_votes, axis=0)

    # Metrics
    overall_acc = (y_true == y_pred).mean()
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)
    cls_acc = per_class_accuracy(cm)
    macro_acc = float(np.mean(cls_acc))

    print("\n=== 7-Model Majority Voting : Test Results ===")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print("Per-class Accuracy:")
    for i, a in enumerate(cls_acc):
        cname = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
        print(f"  {i} ({cname}): {a*100:.2f}%")
    print(f"Macro-average Accuracy: {macro_acc*100:.2f}%")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Save artifacts
    pd.DataFrame(cm,
        index=[f"true_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES]
    ).to_csv(CM_CSV_PATH, index=True)
    print(f"\nSaved confusion matrix to: {CM_CSV_PATH}")

    if SAVE_PREDICTIONS_CSV:
        df = pd.DataFrame({
            "image": all_names,
            "true_label_idx": y_true,
            "pred_label_idx": y_pred,
            "true_label_name": [CLASS_NAMES[i] for i in y_true],
            "pred_label_name": [CLASS_NAMES[i] for i in y_pred],
            "votes_meningioma": votes_all[:, 0],
            "votes_glioma":     votes_all[:, 1],
            "votes_pituitary":  votes_all[:, 2],
            "votes_no_tumor":   votes_all[:, 3],
            "avgp_meningioma":  avg_probs_all[:, 0],
            "avgp_glioma":      avg_probs_all[:, 1],
            "avgp_pituitary":   avg_probs_all[:, 2],
            "avgp_no_tumor":    avg_probs_all[:, 3],
        })
        df.to_csv(PREDICTIONS_CSV_PATH, index=False)
        print(f"Saved per-image predictions to: {PREDICTIONS_CSV_PATH}")

if __name__ == "__main__":
    main()
