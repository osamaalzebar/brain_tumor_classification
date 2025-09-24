#!/usr/bin/env python3
# test_ensemble_5models_customdense.py
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
from torchvision.models import densenet201, DenseNet201_Weights

# ----- Label mapping (CSV 1..4 -> indices 0..3)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

# ----- Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.4815, 0.4578, 0.4082]
CLIP_STD  = [0.2686, 0.2613, 0.2758]

# ----- Your InceptionV3 concat head (import from your training file)
from train_inceptionV3 import InceptionV3ConcatHead  # expects num_classes=4

# ----- RAM builder (same API you used)
from ram.ram.models import ram as build_ram


# =========================
# Dataset: produce 5 inputs per image
# =========================
class TestFiveView(Dataset):
    """
    Emits:
      - incv3: 299, ImageNet
      - res50: 224, ImageNet
      - ram:   384, CLIP (or ImageNet if --ram-imagenet-norm)
      - dense: 224, ImageNet
      - vgg16: 224, ImageNet
    CSV must have headers 'Image_path' and 'label' with labels in {1,2,3,4}.
    """
    def __init__(self, image_root: str, csv_path: str, ram_imagenet_norm: bool = False):
        import os
        self.samples: List[Tuple[str, int]] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
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

        self.tf_incv3 = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.tf_224_imnet = transforms.Compose([
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
        x_inc  = self.tf_incv3(img)
        x_res  = self.tf_224_imnet(img)
        x_ram  = self.tf_ram(img)
        x_den  = self.tf_224_imnet(img)
        x_vgg  = self.tf_224_imnet(img)
        return x_inc, x_res, x_ram, x_den, x_vgg, torch.tensor(y, dtype=torch.long), path


# =========================
# DenseNet201 (custom multi-branch) — same as your working code
# =========================
class MultiBranchDenseNet201(nn.Module):
    def __init__(self, num_classes=4, branch_dim=256, dropout_p=0.5, pretrained=False):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        base = densenet201(weights=weights)
        feats = base.features
        self.stem = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)
        self.db1 = feats.denseblock1; self.tr1 = feats.transition1
        self.db2 = feats.denseblock2; self.tr2 = feats.transition2
        self.db3 = feats.denseblock3; self.tr3 = feats.transition3
        self.db4 = feats.denseblock4; self.out_norm = feats.norm5

        ch_db2 = 512
        ch_db3 = 1792
        ch_db4 = 1920

        def head(in_ch):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=dropout_p),
                nn.Linear(in_ch, branch_dim),
                nn.ReLU(inplace=True),
            )
        self.head2 = head(ch_db2)
        self.head3 = head(ch_db3)
        self.head4 = head(ch_db4)
        self.classifier = nn.Linear(branch_dim * 3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.db1(x); x = self.tr1(x)

        x = self.db2(x); tap2 = x; x = self.tr2(x)
        x = self.db3(x); tap3 = x; x = self.tr3(x)
        x = self.db4(x); tap4 = self.out_norm(x)

        z2 = self.head2(tap2)
        z3 = self.head3(tap3)
        z4 = self.head4(tap4)
        z = torch.cat([z2, z3, z4], dim=1)
        return self.classifier(z)


# =========================
# Other model builders
# =========================
def build_resnet50(num_classes=4):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_vgg16(num_classes=4):
    m = models.vgg16(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m

class RAMWithClassifier(nn.Module):
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

        return self.classifier(torch.cat([feat2, feat3, feat4], dim=1))


# =========================
# Utilities
# =========================
def load_state_dict_flex(model: nn.Module, ckpt_obj):
    """
    Accepts:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}  <-- your VGG16 checkpoint format
      - handles DataParallel 'module.' prefixes
    """
    sd = None
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            sd = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            sd = ckpt_obj["model_state_dict"]

    if sd is None:
        # assume it's already a raw state_dict
        sd = ckpt_obj

    # strip DataParallel 'module.' prefix if present
    sd = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

    # load (use strict=True if you want it to error on mismatches)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("[load_state_dict_flex] Warning:")
        if missing:
            print("  Missing keys (first 10):", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("  Unexpected keys (first 10):", unexpected[:10], "..." if len(unexpected) > 10 else "")

def vote_with_tiebreak(logits_list: List[torch.Tensor]) -> Tuple[int, torch.Tensor]:
    """Hard majority vote; tie-break by highest mean softmax."""
    probs = [F.softmax(l, dim=-1) for l in logits_list]
    preds = [int(torch.argmax(p)) for p in probs]
    votes = torch.bincount(torch.tensor(preds), minlength=probs[0].numel())
    cands = (votes == votes.max()).nonzero(as_tuple=False).flatten().tolist()
    mean_probs = torch.stack(probs, dim=0).mean(dim=0)
    final_idx = cands[0] if len(cands) == 1 else int(cands[torch.argmax(mean_probs[cands])])
    return final_idx, mean_probs


# =========================
# Main
# =========================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Ensemble test: InceptionV3 + ResNet50 + RAM + DenseNet201(custom) + VGG16 (majority vote)")
    ap.add_argument("--image-root", type=str, default=r"D:\brain tumor\classification_task\test\data")
    ap.add_argument("--csv", type=str, default=r"D:\brain tumor\classification_task\test\image_labels_test.csv")

    ap.add_argument("--ckpt-incv3", type=str, default=r'D:\brain tumor\fuse_2\checkpoints\best_incv3_concat.pth', help="Path to best_incv3_concat.pth")
    ap.add_argument("--ckpt-res50", type=str, default=r'D:\brain tumor\fuse_2\checkpoints\best_resnet50.pth', help="Path to best_resnet50.pth")
    ap.add_argument("--ckpt-ram",   type=str, default=r'D:\brain tumor\fuse_2\checkpoints\ram_finetuned_brain_tumor_best.pth')
    ap.add_argument("--ckpt-dense", type=str,default= r'D:\brain tumor\fuse_2\outputs_densenet201\best_model.pt',
                    help=".pt/.pth of your custom MultiBranchDenseNet201")
    ap.add_argument("--ckpt-vgg16", type=str, default= r'D:\brain tumor\fuse_2\checkpoints\best_vgg16_brain.pth')

    ap.add_argument("--ram-pretrained", type=str,default= r'D:\brain tumor\brain_tumor_pyramid_brisc_augment_no_5_fold\checkpoint\ram_swin_large_14m.pth',
                    help="Pretrained backbone for RAM (same file used during RAM training)")
    ap.add_argument("--ram-imagenet-norm", action="store_true",
                    help="Use ImageNet normalization for RAM (default: CLIP norm)")

    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", type=str, default="ensemble5_predictions.csv")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Dataset / Loader
    ds = TestFiveView(args.image_root, args.csv, ram_imagenet_norm=args.ram_imagenet_norm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # ----- Build / load models -----
    # InceptionV3 concat head
    incv3 = InceptionV3ConcatHead(num_classes=4).to(device)
    load_state_dict_flex(incv3, torch.load(args.ckpt_incv3, map_location=device))
    incv3.eval()

    # ResNet50
    res50 = build_resnet50(num_classes=4).to(device)
    load_state_dict_flex(res50, torch.load(args.ckpt_res50, map_location=device))
    res50.eval()

    # RAM (+ pretrained backbone)
    base_ram = build_ram(pretrained=args.ram_pretrained, vit='swin_l', image_size=384).to(device)
    ram = RAMWithClassifier(base_ram, num_classes=4).to(device)
    load_state_dict_flex(ram, torch.load(args.ckpt_ram, map_location=device))
    ram.eval()

    # DenseNet201 (custom multi-branch)
    dense = MultiBranchDenseNet201(num_classes=4, pretrained=False).to(device)
    load_state_dict_flex(dense, torch.load(args.ckpt_dense, map_location=device))
    dense.eval()

    # VGG16
    vgg16 = build_vgg16(num_classes=4).to(device)
    load_state_dict_flex(vgg16, torch.load(args.ckpt_vgg16, map_location=device))
    vgg16.eval()

    # ----- Eval -----
    total = 0
    correct_inc = correct_res = correct_ram = correct_den = correct_vgg = correct_ens = 0
    rows = []

    for x_inc, x_res, x_ram, x_den, x_vgg, y, paths in dl:
        x_inc = x_inc.to(device, non_blocking=True)
        x_res = x_res.to(device, non_blocking=True)
        x_ram = x_ram.to(device, non_blocking=True)
        x_den = x_den.to(device, non_blocking=True)
        x_vgg = x_vgg.to(device, non_blocking=True)
        y     = y.to(device, non_blocking=True)

        l_inc = incv3(x_inc)
        l_res = res50(x_res)
        l_ram = ram(x_ram)
        l_den = dense(x_den)
        l_vgg = vgg16(x_vgg)

        p_inc = l_inc.argmax(dim=1)
        p_res = l_res.argmax(dim=1)
        p_ram = l_ram.argmax(dim=1)
        p_den = l_den.argmax(dim=1)
        p_vgg = l_vgg.argmax(dim=1)

        final_preds = []
        mean_probs_batch = []
        for i in range(y.size(0)):
            final_i, mean_probs_i = vote_with_tiebreak([l_inc[i], l_res[i], l_ram[i], l_den[i], l_vgg[i]])
            final_preds.append(final_i)
            mean_probs_batch.append(mean_probs_i.unsqueeze(0))
        final_preds = torch.tensor(final_preds, device=device)
        mean_probs_batch = torch.cat(mean_probs_batch, dim=0)

        bs = y.size(0)
        total += bs
        correct_inc += (p_inc == y).sum().item()
        correct_res += (p_res == y).sum().item()
        correct_ram += (p_ram == y).sum().item()
        correct_den += (p_den == y).sum().item()
        correct_vgg += (p_vgg == y).sum().item()
        correct_ens += (final_preds == y).sum().item()

        for i in range(bs):
            rows.append({
                "image": paths[i],
                "true_idx": int(y[i].item()),
                "pred_incv3": int(p_inc[i].item()),
                "pred_res50": int(p_res[i].item()),
                "pred_ram": int(p_ram[i].item()),
                "pred_dense": int(p_den[i].item()),
                "pred_vgg16": int(p_vgg[i].item()),
                "pred_ensemble": int(final_preds[i].item()),
                "prob_mean_c0": float(mean_probs_batch[i, 0].item()),
                "prob_mean_c1": float(mean_probs_batch[i, 1].item()),
                "prob_mean_c2": float(mean_probs_batch[i, 2].item()),
                "prob_mean_c3": float(mean_probs_batch[i, 3].item()),
            })

    # Accuracies
    acc_inc = correct_inc / total if total else 0.0
    acc_res = correct_res / total if total else 0.0
    acc_ram = correct_ram / total if total else 0.0
    acc_den = correct_den / total if total else 0.0
    acc_vgg = correct_vgg / total if total else 0.0
    acc_ens = correct_ens / total if total else 0.0

    # Save CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["image","true_idx","pred_ensemble"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved predictions to: {out.resolve()}")
    print(f"Total samples: {total}")
    print(f"InceptionV3 accuracy: {acc_inc:.4f}")
    print(f"ResNet50    accuracy: {acc_res:.4f}")
    print(f"RAM         accuracy: {acc_ram:.4f}")
    print(f"DenseNet201 accuracy: {acc_den:.4f}")
    print(f"VGG16       accuracy: {acc_vgg:.4f}")
    print(f"Ensemble(5) accuracy: {acc_ens:.4f}")


if __name__ == "__main__":
    main()

