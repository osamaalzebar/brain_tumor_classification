#!/usr/bin/env python3
# test_ensemble_incv3_resnet50_ram.py
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms, models

# ---------- Label mapping (CSV 1..4 -> 0..3), per your datasets ----------
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

# ---------- Normalizations & sizes to match each model's training ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD  = [0.2686, 0.2613, 0.2758]

# ---------- Import your Inception-V3 concat head from your training code ----------
# Make sure this import path matches your file (e.g., train_inceptionV3.py)
from train_inceptionV3 import InceptionV3ConcatHead  # expects num_classes=4

# ---------- RAM base builder (same API you used in RAM training) ----------
# If your RAM library lives elsewhere, adjust this import accordingly.
from ram.ram.models import ram as build_ram


# =========================
# Dataset: 3-view pipeline
# =========================
class TestEnsembleDataset(Dataset):
    """
    Loads one CSV and returns three versions of the same image:
      - x_incv3: 299x299, ImageNet norm
      - x_res50: 224x224, ImageNet norm
      - x_ram:   384x384, CLIP norm  (toggle to ImageNet via --ram-imagenet-norm)
    CSV must have headers: "Image_path" and "label" with labels in {1,2,3,4}.
    """
    def __init__(self, image_root: str, csv_path: str, ram_imagenet_norm: bool = False):
        import csv as _csv, os
        self.samples: List[Tuple[str, int]] = []
        with open(csv_path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            # tolerant to "Image_path, label" vs "Image_path,label"
            field_map = {k.strip(): k for k in reader.fieldnames}
            ip = field_map.get("Image_path")
            lb = field_map.get("label")
            if not ip or not lb:
                raise ValueError("CSV must have headers 'Image_path' and 'label'.")
            for row in reader:
                rel = row[ip].strip()
                y_raw = int(row[lb])
                if y_raw not in LABEL_MAP:
                    raise ValueError(f"Unexpected label {y_raw} for {rel}")
                self.samples.append((os.path.join(image_root, rel), LABEL_MAP[y_raw]))

        # Transforms to match your three trainings
        self.tf_incv3 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.tf_res50 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        if ram_imagenet_norm:
            ram_mean, ram_std = IMAGENET_MEAN, IMAGENET_STD
        else:
            ram_mean, ram_std = CLIP_MEAN, CLIP_STD

        self.tf_ram = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(ram_mean, ram_std),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x_incv3 = self.tf_incv3(img)
        x_res50 = self.tf_res50(img)
        x_ram   = self.tf_ram(img)
        return x_incv3, x_res50, x_ram, torch.tensor(y, dtype=torch.long), path


# =========================
# Models
# =========================

def build_resnet50(num_classes=4):
    m = models.resnet50(weights=None)  # we load your fine-tuned state_dict next
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

class RAMWithClassifier(nn.Module):
    """
    Wrap your RAM visual encoder and attach the same head you used in training.
    This example pools stage2/3/4 features (adjust embed_dims to match your training).
    """
    def __init__(self, ram_model, embed_dims=[768, 1536, 6144], dropout=0.3, num_classes=4):
        super().__init__()
        self.ram = ram_model
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.branch2 = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dims[0], 256), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dims[1], 256), nn.ReLU())
        self.branch4 = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dims[2], 256), nn.ReLU())
        self.classifier = nn.Linear(256 * 3, num_classes)

    def forward(self, image):
        B = image.size(0)
        x = self.ram.visual_encoder.patch_embed(image)
        x = self.ram.visual_encoder.pos_drop(x)

        x = self.ram.visual_encoder.layers[0](x)          # stage1
        x2 = self.ram.visual_encoder.layers[1](x)          # stage2
        feat2 = x2.permute(0, 2, 1).reshape(B, -1, 24, 24)
        feat2 = self.pool(feat2).squeeze(-1).squeeze(-1)
        feat2 = self.branch2(feat2)

        x3 = self.ram.visual_encoder.layers[2](x2)         # stage3
        feat3 = x3.permute(0, 2, 1).reshape(B, -1, 12, 12)
        feat3 = self.pool(feat3).squeeze(-1).squeeze(-1)
        feat3 = self.branch3(feat3)

        x4 = self.ram.visual_encoder.layers[3](x3)         # stage4
        feat4 = x4.permute(0, 2, 1).reshape(B, -1, 6, 6)
        feat4 = self.pool(feat4).squeeze(-1).squeeze(-1)
        feat4 = self.branch4(feat4)

        concat_feat = torch.cat([feat2, feat3, feat4], dim=1)
        logits = self.classifier(concat_feat)
        return logits


# =========================
# Voting
# =========================
def vote_with_tiebreak(logits_list: List[torch.Tensor]) -> Tuple[int, torch.Tensor]:
    """Hard majority vote; tie-break by highest mean softmax."""
    probs = [F.softmax(l, dim=-1) for l in logits_list]
    preds = [int(torch.argmax(p)) for p in probs]

    votes = torch.bincount(torch.tensor(preds), minlength=probs[0].numel())
    top = votes.max().item()
    cands = (votes == top).nonzero(as_tuple=False).flatten().tolist()

    mean_probs = torch.stack(probs, dim=0).mean(dim=0)
    final_idx = cands[0] if len(cands) == 1 else int(cands[torch.argmax(mean_probs[cands])])
    return final_idx, mean_probs


# =========================
# Main
# =========================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Ensemble test: InceptionV3 + ResNet50 + RAM (majority vote)")
    ap.add_argument("--image-root", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/data")
    ap.add_argument("--csv", type=str, default= '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/bangladesh_mri/Raw/image_labels.csv')

    ap.add_argument("--ckpt-incv3", type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse _2/checkpoints/best_incv3_concat.pth', help="Path to best_incv3_concat.pth")
    ap.add_argument("--ckpt-res50", type=str, default= '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse _2/checkpoints/best_resnet50.pth', help="Path to best_resnet50.pth")
    ap.add_argument("--ckpt-ram",   type=str, default='/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse _2/checkpoints/ram_finetuned_brain_tumor_best.pth', help="Path to best RAM head/weights (state_dict)")

    ap.add_argument("--ram-pretrained", type=str, default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse _2/checkpoint/ram_swin_large_14m.pth",
                    help="If RAM builder needs a base pretrained file (e.g., swin_l_14M), pass path")
    ap.add_argument("--ram-imagenet-norm", action="store_true",
                    help="Use ImageNet normalization for RAM (default: CLIP norm)")

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", type=str, default="ensemble_predictions.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Dataset/loader (one CSV, 3 pipelines)
    ds = TestEnsembleDataset(args.image_root, args.csv, ram_imagenet_norm=args.ram_imagenet_norm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ----- Build / load models -----
    # Inception-V3 concat head
    incv3 = InceptionV3ConcatHead(num_classes=4).to(device)
    inc_ckpt = torch.load(args.ckpt_incv3, map_location=device)
    incv3.load_state_dict(inc_ckpt["state_dict"])
    incv3.eval()

    # ResNet-50 simple head
    res50 = build_resnet50(num_classes=4).to(device)
    res_ckpt = torch.load(args.ckpt_res50, map_location=device)
    res50.load_state_dict(res_ckpt["state_dict"])
    res50.eval()

    # RAM
    base_ram = build_ram(pretrained=args.ram_pretrained, vit='swin_l', image_size=384).to(device)
    ram = RAMWithClassifier(base_ram, num_classes=4).to(device)
    ram_ckpt = torch.load(args.ckpt_ram, map_location=device)
    # allow both {"state_dict": ...} or raw state_dict
    if isinstance(ram_ckpt, dict) and "state_dict" in ram_ckpt and not any(k.startswith("ram") for k in ram_ckpt.keys()):
        ram.load_state_dict(ram_ckpt["state_dict"])
    else:
        ram.load_state_dict(ram_ckpt)
    ram.eval()

    # ----- Eval loop -----
    total = 0
    correct_incv3 = correct_res50 = correct_ram = correct_ens = 0
    rows = []

    for x_incv3, x_res50, x_ram, y, paths in dl:
        x_incv3 = x_incv3.to(device, non_blocking=True)
        x_res50 = x_res50.to(device, non_blocking=True)
        x_ram   = x_ram.to(device, non_blocking=True)
        y       = y.to(device, non_blocking=True)

        logits_incv3 = incv3(x_incv3)
        logits_res50 = res50(x_res50)
        logits_ram   = ram(x_ram)

        pred_incv3 = logits_incv3.argmax(dim=1)
        pred_res50 = logits_res50.argmax(dim=1)
        pred_ram   = logits_ram.argmax(dim=1)

        final_preds = []
        mean_probs_batch = []
        for i in range(y.size(0)):
            final_i, mean_probs_i = vote_with_tiebreak([
                logits_incv3[i], logits_res50[i], logits_ram[i]
            ])
            final_preds.append(final_i)
            mean_probs_batch.append(mean_probs_i.unsqueeze(0))
        final_preds = torch.tensor(final_preds, device=device)
        mean_probs_batch = torch.cat(mean_probs_batch, dim=0)

        # accuracy bookkeeping
        bs = y.size(0)
        total += bs
        correct_incv3 += (pred_incv3 == y).sum().item()
        correct_res50 += (pred_res50 == y).sum().item()
        correct_ram   += (pred_ram   == y).sum().item()
        correct_ens   += (final_preds == y).sum().item()

        # collect output rows
        for i in range(bs):
            rows.append({
                "image": paths[i],
                "true_idx": int(y[i].item()),
                "pred_incv3": int(pred_incv3[i].item()),
                "pred_res50": int(pred_res50[i].item()),
                "pred_ram": int(pred_ram[i].item()),
                "pred_ensemble": int(final_preds[i].item()),
                "prob_mean_c0": float(mean_probs_batch[i, 0].item()),
                "prob_mean_c1": float(mean_probs_batch[i, 1].item()),
                "prob_mean_c2": float(mean_probs_batch[i, 2].item()),
                "prob_mean_c3": float(mean_probs_batch[i, 3].item()),
            })

    # Accuracies
    acc_incv3 = correct_incv3 / total if total else 0.0
    acc_res50 = correct_res50 / total if total else 0.0
    acc_ram   = correct_ram   / total if total else 0.0
    acc_ens   = correct_ens   / total if total else 0.0

    # Save predictions CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["image","true_idx","pred_ensemble"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved predictions to: {out.resolve()}")
    print(f"Total samples: {total}")
    print(f"InceptionV3 accuracy: {acc_incv3:.4f}")
    print(f"ResNet50   accuracy: {acc_res50:.4f}")
    print(f"RAM        accuracy: {acc_ram:.4f}")
    print(f"Ensemble   accuracy: {acc_ens:.4f}")


if __name__ == "__main__":
    main()

