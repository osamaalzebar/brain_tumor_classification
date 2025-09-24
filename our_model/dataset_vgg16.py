#!/usr/bin/env python3
# dataset_brain_cls.py
from typing import Callable, List, Tuple
import os, csv
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class Rotate90:
    def __init__(self, k: int = 1):
        assert k in (0, 1, 2, 3), "k must be 0,1,2,3"
        self.k = k

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.k == 0:
            return img
        elif self.k == 1:
            return img.transpose(Image.ROTATE_90)
        elif self.k == 2:
            return img.transpose(Image.ROTATE_180)
        else:
            return img.transpose(Image.ROTATE_270)


class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = torch.randint(low=0, high=4, size=(1,)).item()
        return Rotate90(k)(img)


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


class BrainMRIDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, img_size: int = 224, train: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.transform = build_transforms(img_size=img_size, train=train)
        self.samples: List[Tuple[str, int]] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "Image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must have headers: Image_path,label")
            for row in reader:
                rel = row["Image_path"].strip()
                lbl = int(row["label"]) - 1
                path = os.path.join(root_dir, rel)
                self.samples.append((path, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)
