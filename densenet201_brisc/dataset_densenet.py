import os
from PIL import Image
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
import random



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Label mapping (CSV label → model index)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}



class RandomRotate90:
    """Custom transform to rotate the image by 0°, 90°, 180°, or 270° randomly."""
    def __call__(self, img):
        k = random.randint(0, 3)  # 0 to 3 rotations
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

class MRIDataset(Dataset):
    """
    image_dir: folder with images
    csv_file: CSV with Image_path,label (label in {1,2,3,4})
    """
    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        transform: Optional[Callable] = None,
        strict_exists: bool = True,
    ):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        missing_cols = {"Image_path", "label"} - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing columns: {missing_cols}")
        self.transform = transform
        self.strict_exists = strict_exists

        self.samples = []
        for _, row in self.df.iterrows():
            img_name = str(row["Image_path"])
            full_path = os.path.join(self.image_dir, img_name)
            if strict_exists and not os.path.exists(full_path):
                raise FileNotFoundError(f"Image not found: {full_path}")
            y_raw = int(row["label"])
            if y_raw not in LABEL_MAP:
                raise ValueError(f"Unexpected label {y_raw} for {img_name}")
            y = LABEL_MAP[y_raw]
            self.samples.append((full_path, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)
