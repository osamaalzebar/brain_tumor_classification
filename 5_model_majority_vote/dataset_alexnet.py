# dataset_alexnet.py
from pathlib import Path
import csv
import random
from typing import List, Tuple, Callable
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Try to pull mean/std from torchvision weights (works across versions);
# otherwise fall back to standard ImageNet stats.
try:
    from torchvision.models import AlexNet_Weights
    _W = AlexNet_Weights.IMAGENET1K_V1
except Exception:
    _W = None

def _resolve_imagenet_stats():
    if _W is not None:
        meta = getattr(_W, "meta", None)
        if isinstance(meta, dict):
            m, s = meta.get("mean"), meta.get("std")
            if m is not None and s is not None:
                return tuple(m), tuple(s)
        try:
            t = _W.transforms()
            m, s = getattr(t, "mean", None), getattr(t, "std", None)
            if m is not None and s is not None:
                return tuple(m), tuple(s)
        except Exception:
            pass
    # Fallback
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

IMAGENET_MEAN, IMAGENET_STD = _resolve_imagenet_stats()

CLASS_NAMES = ["meningioma", "glioma", "pituitary", "no_tumor"]

class RandomRotate90:
    """Rotate image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img.rotate(90 * k)

def build_transforms(img_size: int = 224, train: bool = True) -> Callable:
    # AlexNet expects 3-channel input. Your MRIs are grayscale,
    # so we convert to 3 channels before normalization.
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

class MRIBrainTumorCSV(Dataset):
    """
    CSV header: Image_path,label
    - Image_path: file name under 'root_dir'
    - label: 1..4 (1=meningioma, 2=glioma, 3=pituitary, 4=no_tumor)
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
                label_idx = label_raw - 1  # map 1..4 -> 0..3
                self.samples.append((self.root_dir / img_name, label_idx))

        if not self.samples:
            raise RuntimeError(f"No samples found from CSV: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            # ensure grayscale read is robust, but transforms.Grayscale(3) handles it anyway
            img = img.convert("L")
            if self.transform:
                img = self.transform(img)
        return img, label
