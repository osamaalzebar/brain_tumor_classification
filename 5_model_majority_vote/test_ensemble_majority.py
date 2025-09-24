
import argparse
import csv
import os
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import (
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    alexnet, AlexNet_Weights,
    googlenet, GoogLeNet_Weights,
)
from torchvision import transforms

import pretrainedmodels
from PIL import Image

# ---- Robust mean/std helper (handles different torchvision versions) ----
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

def mean_std_from_weights(weights, fallback_mean=_IMAGENET_MEAN, fallback_std=_IMAGENET_STD):
    # Try common places, then fallback
    try:
        meta = getattr(weights, "meta", None)
        if meta and "mean" in meta and "std" in meta:
            return tuple(meta["mean"]), tuple(meta["std"])
    except Exception:
        pass
    try:
        # torchvision >= 0.13 provides weights.transforms()
        tfm = weights.transforms()
        # transforms may be a callable that returns a composition; try to inspect for Normalize
        if hasattr(tfm, "transforms"):
            for t in getattr(tfm, "transforms", []):
                if isinstance(t, transforms.Normalize):
                    return tuple(t.mean), tuple(t.std)
    except Exception:
        pass
    return fallback_mean, fallback_std

# ---------------------------
# Helpers
# ---------------------------

def accuracy(true: List[int], pred: List[int]) -> float:
    total = len(true)
    correct = sum(int(a == b) for a, b in zip(true, pred))
    return (correct / total) if total > 0 else 0.0

def majority_vote_row(votes: List[int]) -> int:
    c = Counter(votes)
    max_count = max(c.values())
    best = [k for k, v in c.items() if v == max_count]
    return min(best)

def read_csv_list(csv_path: str) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        assert "Image_path" in reader.fieldnames and "label" in reader.fieldnames,             "CSV must have header: Image_path,label"
        for row in reader:
            paths.append(row["Image_path"].strip())
            labels.append(int(row["label"]) - 1)  # map 1..4 -> 0..3
    return paths, labels

# ---------------------------
# Dataset (no augmentation)
# ---------------------------

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir: str, csv_path: str, transform):
        self.images_dir = images_dir
        self.fns, self.labels = read_csv_list(csv_path)
        self.transform = transform
        for fn in self.fns:
            p = os.path.join(images_dir, fn)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

    def __len__(self): return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.images_dir, fn)).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ---------------------------
# Transforms (no aug, per-architecture normalization)
# ---------------------------

def tfms_shufflenet_like(img_size: int):
    w = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
    mean, std = mean_std_from_weights(w)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def tfms_squeezenet_like(img_size: int):
    w = SqueezeNet1_1_Weights.IMAGENET1K_V1
    mean, std = mean_std_from_weights(w)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def tfms_alexnet_like(img_size: int):
    w = AlexNet_Weights.IMAGENET1K_V1
    mean, std = mean_std_from_weights(w)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def tfms_googlenet_like(img_size: int):
    w = GoogLeNet_Weights.IMAGENET1K_V1
    mean, std = mean_std_from_weights(w)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def tfms_nasnet_from_pretrainedsettings():
    setting = pretrainedmodels.pretrained_settings['nasnetamobile']['imagenet']
    mean = tuple(setting['mean'])
    std = tuple(setting['std'])
    img_size = setting.get('input_size', (3, 224, 224))[1]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]), img_size

# ---------------------------
# Model builders (match fine-tuned heads)
# ---------------------------

def build_shufflenet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
    model = shufflenet_v2_x1_0(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def build_squeezenet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = squeezenet1_1(weights=weights)
    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    return model

def build_alexnet(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = alexnet(weights=weights)
    in_features = model.classifier[1].in_features  # 9216
    hidden = 4096
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes),
    )
    return model

def build_googlenet(num_classes: int = 4, pretrained: bool = True, aux_logits: bool = True) -> nn.Module:
    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights, aux_logits=aux_logits)
    model.fc = nn.Linear(1024, num_classes)
    if aux_logits:
        model.aux1.fc = nn.Linear(1024, num_classes)
        model.aux2.fc = nn.Linear(1024, num_classes)
    return model

class NASNetMobileClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 4):
        super().__init__()
        self.backbone = backbone
    def forward(self, x):
        return self.backbone(x)

def build_nasnet_mobile(num_classes: int = 4) -> nn.Module:
    base = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained=None)
    in_features = base.last_linear.in_features
    base.last_linear = nn.Linear(in_features, num_classes)
    return NASNetMobileClassifier(base, num_classes=num_classes)

# ---------------------------
# Inference
# ---------------------------

@torch.no_grad()
def model_predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true, y_pred = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        y_true.extend(targets.tolist())
        y_pred.extend(preds.tolist())
    return y_true, y_pred

# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser("Majority-vote ensemble over 5 fine-tuned models (NO test-time augmentation)")
    p.add_argument("--test_dir", type=str, required=False,
                   default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/data")
    p.add_argument("--test_csv", type=str, required=False,
                   default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/image_labels.csv")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--ckpt_shufflenet", type=str, default="checkpoints/shufflenet_v2_brain_tumor_best.pth")
    p.add_argument("--imgsize_shufflenet", type=int, default=224)
    p.add_argument("--no_pretrained_shufflenet", action="store_true")

    p.add_argument("--ckpt_squeezenet", type=str, default="checkpoints/squeezenet_best.pth")
    p.add_argument("--imgsize_squeezenet", type=int, default=224)
    p.add_argument("--no_pretrained_squeezenet", action="store_true")

    p.add_argument("--ckpt_alexnet", type=str, default="checkpoints/alexnet_best.pth")
    p.add_argument("--imgsize_alexnet", type=int, default=224)
    p.add_argument("--no_pretrained_alexnet", action="store_true")

    p.add_argument("--ckpt_googlenet", type=str, default="checkpoints/googlenet_best.pth")
    p.add_argument("--imgsize_googlenet", type=int, default=224)
    p.add_argument("--no_pretrained_googlenet", action="store_true")
    p.add_argument("--no_aux_logits_googlenet", action="store_true")

    p.add_argument("--ckpt_nasnet", type=str, default="checkpoints/best_nasnet_mobile.pth")

    args = p.parse_args()
    device = torch.device(args.device)

    csv_fns, csv_labels = read_csv_list(args.test_csv)
    n_samples = len(csv_fns)

    # ShuffleNet
    tfm = tfms_shufflenet_like(args.imgsize_shufflenet)
    ds = CSVDataset(args.test_dir, args.test_csv, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    m_shuf = build_shufflenet(num_classes=4, pretrained=not args.no_pretrained_shufflenet).to(device)
    st = torch.load(args.ckpt_shufflenet, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    m_shuf.load_state_dict(st, strict=False)
    y_true_shuf, y_pred_shuf = model_predict_logits(m_shuf, dl, device)
    acc_shuf = accuracy(y_true_shuf, y_pred_shuf)
    print(f"[ShuffleNetV2] accuracy: {acc_shuf:.4f} ({sum(int(a==b) for a,b in zip(y_true_shuf,y_pred_shuf))}/{len(y_true_shuf)})")

    # SqueezeNet
    tfm = tfms_squeezenet_like(args.imgsize_squeezenet)
    ds = CSVDataset(args.test_dir, args.test_csv, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    m_sq = build_squeezenet(num_classes=4, pretrained=not args.no_pretrained_squeezenet).to(device)
    st = torch.load(args.ckpt_squeezenet, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    m_sq.load_state_dict(st, strict=False)
    y_true_sq, y_pred_sq = model_predict_logits(m_sq, dl, device)
    acc_sq = accuracy(y_true_sq, y_pred_sq)
    print(f"[SqueezeNet]   accuracy: {acc_sq:.4f} ({sum(int(a==b) for a,b in zip(y_true_sq,y_pred_sq))}/{len(y_true_sq)})")

    # AlexNet
    tfm = tfms_alexnet_like(args.imgsize_alexnet)
    ds = CSVDataset(args.test_dir, args.test_csv, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    m_al = build_alexnet(num_classes=4, pretrained=not args.no_pretrained_alexnet).to(device)
    st = torch.load(args.ckpt_alexnet, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    m_al.load_state_dict(st, strict=False)
    y_true_al, y_pred_al = model_predict_logits(m_al, dl, device)
    acc_al = accuracy(y_true_al, y_pred_al)
    print(f"[AlexNet]      accuracy: {acc_al:.4f} ({sum(int(a==b) for a,b in zip(y_true_al,y_pred_al))}/{len(y_true_al)})")

    # GoogLeNet
    tfm = tfms_googlenet_like(args.imgsize_googlenet)
    ds = CSVDataset(args.test_dir, args.test_csv, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    m_gg = build_googlenet(num_classes=4, pretrained=not args.no_pretrained_googlenet,
                           aux_logits=not args.no_aux_logits_googlenet).to(device)
    st = torch.load(args.ckpt_googlenet, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    m_gg.load_state_dict(st, strict=False)
    y_true_gg, y_pred_gg = model_predict_logits(m_gg, dl, device)
    acc_gg = accuracy(y_true_gg, y_pred_gg)
    print(f"[GoogLeNet]    accuracy: {acc_gg:.4f} ({sum(int(a==b) for a,b in zip(y_true_gg,y_pred_gg))}/{len(y_true_gg)})")

    # NASNet-Mobile
    tfm_nas, nas_img_size = tfms_nasnet_from_pretrainedsettings()
    ds = CSVDataset(args.test_dir, args.test_csv, transform=tfm_nas)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    m_ns = build_nasnet_mobile(num_classes=4).to(device)
    if not os.path.isfile(args.ckpt_nasnet):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_nasnet}")
    st = torch.load(args.ckpt_nasnet, map_location="cpu")
    st = st.get("model", st)
    m_ns.load_state_dict(st, strict=False)
    y_true_ns, y_pred_ns = model_predict_logits(m_ns, dl, device)
    acc_ns = accuracy(y_true_ns, y_pred_ns)
    print(f"[NASNetMobile] accuracy: {acc_ns:.4f} ({sum(int(a==b) for a,b in zip(y_true_ns,y_pred_ns))}/{len(y_true_ns)})")

    # Sanity checks
    assert len(y_true_shuf)==len(y_true_sq)==len(y_true_al)==len(y_true_gg)==len(y_true_ns)==n_samples,         "Mismatch in #samples across loaders"
    assert y_true_shuf == y_true_sq == y_true_al == y_true_gg == y_true_ns == csv_labels,         "Label order mismatch across datasets; ensure all use the SAME CSV order without shuffling."

    # Majority vote
    ensemble_preds = []
    for i in range(n_samples):
        votes = [y_pred_shuf[i], y_pred_sq[i], y_pred_al[i], y_pred_gg[i], y_pred_ns[i]]
        ensemble_preds.append(majority_vote_row(votes))

    ens_acc = accuracy(csv_labels, ensemble_preds)
    print("\n=== Ensemble (Majority Vote over 5 models) ===")
    print(f"Ensemble accuracy: {ens_acc:.4f} ({sum(int(a==b) for a,b in zip(csv_labels,ensemble_preds))}/{n_samples})")

if __name__ == "__main__":
    main()