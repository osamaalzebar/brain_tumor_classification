import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random

class RandomRotate90:
    """Custom transform to rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)  # 0 to 3 rotations
        return img.rotate(90 * k)

class BrainMRIDataset(Dataset):
    def __init__(self, df, image_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, label = self.df.iloc[idx]
        full_path = os.path.join(self.image_root, image_path)

        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # CSV labels are 1..4; convert to 0..3 for CrossEntropyLoss
        return image, int(label) - 1


def get_transforms(augment=False):
    """
    augment=False → no augmentation (for validation/testing)
    augment=True  → apply horizontal flip + random 90° rotation
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758])
        ])



