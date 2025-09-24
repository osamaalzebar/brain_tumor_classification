#!/usr/bin/env python3
# dataset_brain_csv.py
import csv
import os
import random
from typing import Callable, List, Tuple
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Map CSV labels {1..4} -> {0..3} indices
# 1: meningioma, 2: glioma, 3: pituitary, 4: no tumor
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

class RandomRotate90:
    """Rotate by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)  # 0..3
        return img.rotate(90 * k)

def build_transforms(img_size: int = 299, train: bool = True) -> Callable:
    # Inception-v3 is usually trained with 299x299
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

class BrainCSVSet(Dataset):
    """
    CSV format:
      header: "Image_path, label"   (note: may include a space after the comma)
      rows:   <filename>,<int_label in {1,2,3,4}>
    image_root points to the 'data' folder that actually contains those files.
    """
    def __init__(self, image_root: str, csv_path: str, train: bool, img_size: int = 299):
        self.image_root = str(image_root)
        self.csv_path   = str(csv_path)
        self.items: List[Tuple[str, int]] = []

        # robust header handling ("Image_path, label" vs "Image_path,label")
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            # normalize field names by stripping spaces
            field_map = {name.strip(): name for name in reader.fieldnames}
            ip_key = field_map.get("Image_path")
            lb_key = field_map.get("label")
            if ip_key is None or lb_key is None:
                raise ValueError("CSV must have headers 'Image_path' and 'label'.")

            for row in reader:
                rel = row[ip_key].strip()
                y_raw = int(row[lb_key])
                if y_raw not in LABEL_MAP:
                    raise ValueError(f"Unexpected label {y_raw} for row {row}")
                y = LABEL_MAP[y_raw]
                self.items.append((os.path.join(self.image_root, rel), y))

        self.tf = build_transforms(img_size=img_size, train=train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.tf(img)
        return img, torch.tensor(y, dtype=torch.long), path

