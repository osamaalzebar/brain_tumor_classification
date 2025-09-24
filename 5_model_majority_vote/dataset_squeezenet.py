# dataset_squeezenet.py
from pathlib import Path
import csv
import random
from typing import List, Tuple, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Optional import guards: older torchvision may lack this Enum
try:
    from torchvision.models import SqueezeNet1_1_Weights
    _WEIGHTS = SqueezeNet1_1_Weights.IMAGENET1K_V1
except Exception:
    _WEIGHTS = None

# --- Resolve proper mean/std robustly across torchvision versions ---
def _resolve_imagenet_stats():
    # 1) Try weights.meta (newer versions)
    if _WEIGHTS is not None:
        meta = getattr(_WEIGHTS, "meta", None)
        if isinstance(meta, dict):
            mean = meta.get("mean", None)
            std  = meta.get("std", None)
            if mean is not None and std is not None:
                return tuple(mean), tuple(std)

        # 2) Try weights.transforms() (some versions expose mean/std on the preset)
        try:
            t = _WEIGHTS.transforms()
            mean = getattr(t, "mean", None)
            std  = getattr(t, "std", None)
            if mean is not None and std is not None:
                return tuple(mean), tuple(std)
        except Exception:
            pass

    # 3) Fallback to standard ImageNet stats
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

IMAGENET_MEAN, IMAGENET_STD = _resolve_imagenet_stats()

# --- Class names (for logs) ---
CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]

class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)  # 0..3
        return img.rotate(90 * k)

def build_transforms(img_size: int = 224, train: bool = True) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class MRIBrainTumorCSV(Dataset):
    """
    CSV header: Image_path,label
    - Image_path: file name under 'root_dir'
    - label: integers 1..4 (1=meningioma, 2=glioma, 3=pituitary, 4=no tumor)
    """
    def __init__(self, root_dir: str, csv_path: str, transform=None):
        self.root_dir = Path(root_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header
            for row in reader:
                if not row or len(row) < 2:
                    continue
                img_name = row[0].strip()
                label_raw = int(row[1])
                label_idx = label_raw - 1  # 1..4 -> 0..3
                self.samples.append((self.root_dir / img_name, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found from CSV: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path).convert("RGB") as img:
            if self.transform:
                img = self.transform(img)
        return img, label
