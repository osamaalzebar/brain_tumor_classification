import csv
import os
import random
from typing import Callable, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# We’ll read mean/std from the model’s pretrained cfg in the train script and pass them in.
# Keep these as placeholders in case someone imports this without passing stats:
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD  = (0.229, 0.224, 0.225)

IMAGENET_MEAN = DEFAULT_MEAN
IMAGENET_STD = DEFAULT_STD


class RandomRotate90:
    """Rotate the image by 0°, 90°, 180°, or 270° randomly (PIL.Image.rotate)."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img.rotate(90 * k, expand=True)


def build_transforms(
    img_size: int = 224,
    train: bool = True,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> Callable:
    # NASNet (via timm) expects 3-channel inputs; convert grayscale -> 3-channel before normalization.
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomRotate90(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class BrainTumorCSVDataset(Dataset):
    """
    Expects:
      - images_dir: folder containing images
      - labels_csv: CSV with header: Image_path,label
        * Image_path is just the filename inside images_dir
        * label is 1..4 (1=meningioma, 2=glioma, 3=pituitary, 4=no tumor)
    We convert labels to 0..3 for PyTorch CrossEntropyLoss.
    """
    def __init__(self, images_dir: str, labels_csv: str, transform: Callable = None):
        self.images_dir = images_dir
        self.labels_csv = labels_csv
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        self._load_csv()

    def _load_csv(self):
        if not os.path.isfile(self.labels_csv):
            raise FileNotFoundError(f"CSV not found: {self.labels_csv}")
        if not os.path.isdir(self.images_dir):
            raise NotADirectoryError(f"Images dir not found: {self.images_dir}")

        with open(self.labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "Image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError("CSV must have header: Image_path,label")

            for row in reader:
                rel_path = row["Image_path"].strip()
                label_raw = int(row["label"])
                # Map 1..4 -> 0..3
                label = label_raw - 1
                img_path = os.path.join(self.images_dir, rel_path)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
                self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found from CSV: {self.labels_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # start as grayscale; we’ll expand to 3ch in transform

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
