import os
import math
import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# === Per-model transforms from your training datasets ===
from dataset_RAM import get_transforms as ram_get_transforms
from dataset_densenet import build_transforms as dn_build_transforms

# === DenseNet201 (multi-branch head matching your training) ===
import torchvision
from torchvision.models import densenet201, DenseNet201_Weights

class MultiBranchDenseNet201(nn.Module):
    def __init__(self, num_classes=4, branch_dim=256, dropout_p=0.5, pretrained=True):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        base = densenet201(weights=weights)
        feats = base.features
        self.stem = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)
        self.db1 = feats.denseblock1; self.tr1 = feats.transition1
        self.db2 = feats.denseblock2; self.tr2 = feats.transition2
        self.db3 = feats.denseblock3; self.tr3 = feats.transition3
        self.db4 = feats.denseblock4; self.out_norm = feats.norm5

        ch_db2, ch_db3, ch_db4 = 512, 1792, 1920
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
        z2 = self.head2(tap2); z3 = self.head3(tap3); z4 = self.head4(tap4)
        z = torch.cat([z2, z3, z4], dim=1)
        return self.classifier(z)

# === RAM wrapper (same head as your training) ===
from ram.ram.models import ram  # ensure your RAM repo is on PYTHONPATH

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
        x = self.ram.visual_encoder.layers[0](x)
        x2 = self.ram.visual_encoder.layers[1](x)

        feat2 = x2.permute(0,2,1).reshape(B, -1, 24, 24)
        feat2 = self.pool(feat2).squeeze(-1).squeeze(-1)
        feat2 = self.branch2(feat2)

        x3 = self.ram.visual_encoder.layers[2](x2)
        feat3 = x3.permute(0,2,1).reshape(B, -1, 12, 12)
        feat3 = self.pool(feat3).squeeze(-1).squeeze(-1)
        feat3 = self.branch3(feat3)

        x4 = self.ram.visual_encoder.layers[3](x3)
        feat4 = x4.permute(0,2,1).reshape(B, -1, 6, 6)
        feat4 = self.pool(feat4).squeeze(-1).squeeze(-1)
        feat4 = self.branch4(feat4)

        concat_feat = torch.cat([feat2, feat3, feat4], dim=1)
        return self.classifier(concat_feat)

# === Temperature scaling ===
class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float))
    def forward(self, logits):
        T = torch.exp(self.log_T).clamp(min=1e-3, max=100.0)
        return logits / T

def fit_temperature(model, dataloader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device); labels = labels.to(device)
            logits = model(imgs); all_logits.append(logits); all_labels.append(labels)
    all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)

    def closure():
        optimizer.zero_grad()
        scaled = scaler(all_logits)
        loss = ce(scaled, all_labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return scaler, float(torch.exp(scaler.log_T).item())

# === TTA dataset that uses each model's own normalization/size ===
class TTADatasetPerModel(Dataset):
    """
    CSV must have columns: Image_path,label  (labels 1..4)
    base_transform should already include resize + ToTensor + Normalize for that model.
    """
    def __init__(self, df, image_root, base_transform, expand_tta=False):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.base_transform = base_transform
        self.expand_tta = expand_tta

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["Image_path"]
        label = int(self.df.iloc[idx]["label"]) - 1  # 0..3
        path = os.path.join(self.image_root, img_name)
        img = Image.open(path).convert("RGB")
        if not self.expand_tta:
            return self.base_transform(img), label
        views = [
            img,
            img.transpose(Image.FLIP_LEFT_RIGHT),
            img.rotate(90, expand=True),
            img.rotate(180, expand=True),
            img.rotate(270, expand=True),
        ]
        views_t = [self.base_transform(v) for v in views]
        x = torch.stack(views_t, dim=0)  # [V, C, H, W]
        return x, label

@torch.no_grad()
def logits_with_tta(model, batch_x, device, temp_scaler=None):
    model.eval()
    if batch_x.dim() == 5:
        B, V, C, H, W = batch_x.shape
        logits = model(batch_x.view(B*V, C, H, W).to(device)).view(B, V, -1).mean(1)
    else:
        logits = model(batch_x.to(device))
    if temp_scaler is not None:
        logits = temp_scaler(logits)
    return logits

def accuracy_from_logits(z, y):
    return (z.argmax(1) == y).float().mean().item()

def confusion_matrix(z, y, num_classes=4):
    preds = z.argmax(1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=y.device)
    for t, p in zip(y.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def grid_search_weight(z_ram, z_dn, y, step=0.01):
    best_w, best_acc = 0.0, 0.0
    w = 0.0
    while w <= 1.0001:
        fused = (1 - w) * z_ram + w * z_dn
        acc = accuracy_from_logits(fused, y)
        if acc > best_acc:
            best_acc, best_w = acc, w
        w += step
    return best_w, best_acc

def main():
    ap = argparse.ArgumentParser(description="Decision-level fusion (Calibrated + TTA + Weighted logits)")

    # === Models & checkpoints ===
    ap.add_argument("--ram_pretrained", type=str, required=True, help="RAM pretraining checkpoint (e.g., ram_swin_large_14m.pth)")
    ap.add_argument("--ram_ckpt", type=str, required=True, help="Fine-tuned RAM head checkpoint (your best .pth)")
    ap.add_argument("--dense_ckpt", type=str, required=True, help="Fine-tuned DenseNet201 checkpoint (.pt/.pth)")

    # === Validation for learning fusion params ===
    ap.add_argument("--val_image_root", type=str,
        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/val/data")
    ap.add_argument("--val_csv", type=str,
        default="/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/val/image_lables.csv")

    # === Optional test set (if omitted, reuse validation for evaluation) ===
    ap.add_argument("--test_image_root", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)

    # === Optional pre-saved fusion params ===
    ap.add_argument("--fusion_params", type=str, default=None,
                    help='JSON: {"T_ram":..., "T_dense":..., "w":...}')
    ap.add_argument("--save_params", type=str, default=None,
                    help="Where to save learned fusion params (JSON)")

    # === Inference options ===
    ap.add_argument("--ram_img_size", type=int, default=384)
    ap.add_argument("--dense_img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_tta", action="store_true", help="Disable TTA (single view)")

    # === Optional: save predictions ===
    ap.add_argument("--pred_csv", type=str, default=None, help="Where to save predictions (CSV)")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Build models & load checkpoints ===
    # DenseNet
    dense_model = MultiBranchDenseNet201(num_classes=4, branch_dim=256, dropout_p=0.5, pretrained=True).to(device)
    dckpt = torch.load(args.dense_ckpt, map_location="cpu")
    dense_state = dckpt["state_dict"] if isinstance(dckpt, dict) and "state_dict" in dckpt else dckpt
    dense_model.load_state_dict(dense_state); dense_model.eval()

    # RAM
    base_ram = ram(pretrained=args.ram_pretrained, vit='swin_l', image_size=args.ram_img_size).to(device)
    for p in base_ram.visual_encoder.parameters():
        p.requires_grad = False
    ram_model = RAMWithClassifier(base_ram, num_classes=4).to(device)
    ram_model.load_state_dict(torch.load(args.ram_ckpt, map_location="cpu")); ram_model.eval()

    # === Per-model transforms (match training) ===
    ram_transform   = ram_get_transforms(augment=False)
    dense_transform = dn_build_transforms(img_size=args.dense_img_size, train=False)

    # === Learn / load fusion params on VALIDATION ===
    if args.fusion_params is not None:
        with open(args.fusion_params, "r") as f:
            params = json.load(f)
        T_ram   = float(params["T_ram"]); T_dense = float(params["T_dense"]); w = float(params["w"])
        ram_temp  = TemperatureScaler(T_ram).to(device); ram_temp.log_T.data = torch.log(torch.tensor(T_ram))
        dense_temp = TemperatureScaler(T_dense).to(device); dense_temp.log_T.data = torch.log(torch.tensor(T_dense))
        print(f"[Loaded fusion params] T_RAM={T_ram:.3f} | T_Dense={T_dense:.3f} | w={w:.2f}")
    else:
        if args.val_csv is None or args.val_image_root is None:
            raise ValueError("Need --val_csv and --val_image_root to learn fusion params.")

        val_df = pd.read_csv(args.val_csv)
        if not {"Image_path", "label"}.issubset(val_df.columns):
            raise ValueError("--val_csv must have columns: Image_path,label")

        # Calibration (no TTA)
        ram_val_calib = TTADatasetPerModel(val_df, args.val_image_root, ram_transform,   expand_tta=False)
        dn_val_calib  = TTADatasetPerModel(val_df, args.val_image_root, dense_transform, expand_tta=False)
        ram_calib_loader = DataLoader(ram_val_calib, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
        dn_calib_loader  = DataLoader(dn_val_calib,  batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)

        # Fit temperatures
        ram_temp, T_ram     = fit_temperature(ram_model, ram_calib_loader, device)
        dense_temp, T_dense = fit_temperature(dense_model, dn_calib_loader, device)
        print(f"[Calibration] T_RAM={T_ram:.3f} | T_Dense={T_dense:.3f}")

        # TTA (or single view) for weight search
        expand = not args.no_tta
        ram_val_tta = TTADatasetPerModel(val_df, args.val_image_root, ram_transform,   expand_tta=expand)
        dn_val_tta  = TTADatasetPerModel(val_df, args.val_image_root, dense_transform, expand_tta=expand)
        ram_val_loader = DataLoader(ram_val_tta, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
        dn_val_loader  = DataLoader(dn_val_tta,  batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

        # Collect aligned logits
        all_labels = []
        all_ram_logits, all_dn_logits = [], []
        dn_iter = iter(dn_val_loader)
        with torch.no_grad():
            for x_ram, y in tqdm(ram_val_loader, desc="Val pass for fusion weight"):
                z_ram = logits_with_tta(ram_model, x_ram, device, temp_scaler=ram_temp)
                x_dn, _ = next(dn_iter)
                z_dn = logits_with_tta(dense_model, x_dn, device, temp_scaler=dense_temp)
                all_labels.append(y.to(device))
                all_ram_logits.append(z_ram)
                all_dn_logits.append(z_dn)
        y_val = torch.cat(all_labels, 0)
        z_ram_val = torch.cat(all_ram_logits, 0)
        z_dn_val  = torch.cat(all_dn_logits, 0)

        # Individual accuracies
        acc_ram = accuracy_from_logits(z_ram_val, y_val)
        acc_dn  = accuracy_from_logits(z_dn_val,  y_val)
        print(f"[Val] RAM: {acc_ram*100:.2f}% | DenseNet: {acc_dn*100:.2f}%")

        # Find best DenseNet weight
        w, best_acc = grid_search_weight(z_ram_val, z_dn_val, y_val, step=0.01)
        print(f"[Val Fusion] best w={w:.2f} → Acc={best_acc*100:.2f}%")

        # Optionally save params
        if args.save_params is not None:
            with open(args.save_params, "w") as f:
                json.dump({"T_ram": T_ram, "T_dense": T_dense, "w": w}, f, indent=2)
            print(f"[Saved] fusion params → {args.save_params}")

    # === Build EVAL data (test if provided, else validation) ===
    if args.test_csv is None or args.test_image_root is None:
        print("[Info] No test set provided; evaluating on the validation set.")
        eval_image_root = args.val_image_root
        eval_df = pd.read_csv(args.val_csv)
    else:
        eval_image_root = args.test_image_root
        eval_df = pd.read_csv(args.test_csv)

    if not {"Image_path", "label"}.issubset(eval_df.columns):
        raise ValueError("Eval CSV must have columns: Image_path,label")

    expand = not args.no_tta
    ram_eval_ds = TTADatasetPerModel(eval_df, eval_image_root, ram_transform,   expand_tta=expand)
    dn_eval_ds  = TTADatasetPerModel(eval_df, eval_image_root, dense_transform, expand_tta=expand)
    ram_eval_loader = DataLoader(ram_eval_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
    dn_eval_loader  = DataLoader(dn_eval_ds,  batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

    # === Evaluate ===
    all_logits, all_labels = [], []
    dn_iter = iter(dn_eval_loader)
    with torch.no_grad():
        for x_ram, y in tqdm(ram_eval_loader, desc="EVAL"):
            z_ram = logits_with_tta(ram_model, x_ram, device, temp_scaler=ram_temp)
            x_dn, _ = next(dn_iter)
            z_dn = logits_with_tta(dense_model, x_dn, device, temp_scaler=dense_temp)
            z_fuse = (1 - w) * z_ram + w * z_dn
            all_logits.append(z_fuse)
            all_labels.append(y.to(device))
    z_eval = torch.cat(all_logits, 0)
    y_eval = torch.cat(all_labels, 0)

    acc = accuracy_from_logits(z_eval, y_eval)
    cm = confusion_matrix(z_eval, y_eval, num_classes=4).cpu().numpy()
    print(f"\n[EVAL FUSION] Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    # Optional: save predictions
    if args.pred_csv is not None:
        preds = z_eval.argmax(1).cpu().numpy()
        out = eval_df.copy()
        out["pred_label"] = preds + 1  # back to 1..4
        out.to_csv(args.pred_csv, index=False)
        print(f"[Saved] predictions → {args.pred_csv}")

if __name__ == "__main__":
    main()
