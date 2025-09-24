import os
import math
import json
import argparse
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# === Per-model transforms from your training datasets ===
from dataset_RAM import get_transforms as ram_get_transforms      # RAM: CLIP-like stats, 384
from dataset_densenet import build_transforms as dn_build_transforms  # DenseNet: ImageNet stats, 224

# === DenseNet201 (multi-branch head as in your training) ===
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
from ram.ram.models import ram  # ensure RAM repo is on PYTHONPATH

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

# === Temperature scaling module (used with manual T) ===
class TemperatureScaler(nn.Module):
    def __init__(self, init_T=1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(math.log(max(init_T, 1e-6))), requires_grad=False)
    def forward(self, logits):
        T = torch.exp(self.log_T).clamp(min=1e-3, max=100.0)
        return logits / T

# === TTA dataset that uses each model's own normalization/size ===
class TTADatasetPerModel(Dataset):
    """
    CSV must have columns: Image_path,label  (labels 1..4)
    base_transform should already include resize + ToTensor + Normalize for that model.
    """
    def __init__(self, df, image_root, base_transform, expand_tta=True):
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
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def main():
    ap = argparse.ArgumentParser(description="Test decision-level fusion with MANUAL fusion params")
    # Test data + models
    ap.add_argument("--test_image_root", type=str, required=True)
    ap.add_argument("--test_csv", type=str, required=True)  # columns: Image_path,label
    ap.add_argument("--ram_pretrained", type=str, required=True)
    ap.add_argument("--ram_ckpt", type=str, required=True)
    ap.add_argument("--dense_ckpt", type=str, required=True)

    # Manual fusion params (override everything else if provided)
    ap.add_argument("--w", type=float, default=None, help="Fusion weight for DenseNet logits (0..1).")
    ap.add_argument("--T_ram", type=float, default=None, help="Temperature for RAM logits (>=1e-3).")
    ap.add_argument("--T_dense", type=float, default=None, help="Temperature for DenseNet logits (>=1e-3).")

    # Alternatively load from JSON
    ap.add_argument("--fusion_params", type=str, default=None, help='JSON with {"T_ram":..., "T_dense":..., "w":...}')

    # Inference options
    ap.add_argument("--ram_img_size", type=int, default=384)
    ap.add_argument("--dense_img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_tta", action="store_true", help="Disable TTA (use single view)")
    ap.add_argument("--pred_csv", type=str, default=None, help="Optional: save test predictions to CSV")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Build models & load checkpoints ===
    dense_model = MultiBranchDenseNet201(num_classes=4, branch_dim=256, dropout_p=0.5, pretrained=True).to(device)
    dckpt = torch.load(args.dense_ckpt, map_location="cpu")
    dense_state = dckpt["state_dict"] if isinstance(dckpt, dict) and "state_dict" in dckpt else dckpt
    dense_model.load_state_dict(dense_state); dense_model.eval()

    base_ram = ram(pretrained=args.ram_pretrained, vit='swin_l', image_size=args.ram_img_size).to(device)
    for p in base_ram.visual_encoder.parameters(): p.requires_grad = False
    ram_model = RAMWithClassifier(base_ram, num_classes=4).to(device)
    ram_model.load_state_dict(torch.load(args.ram_ckpt, map_location="cpu")); ram_model.eval()

    # === Transforms (match training)
    ram_transform   = ram_get_transforms(augment=False)
    dense_transform = dn_build_transforms(img_size=args.dense_img_size, train=False)

    # === Load fusion params: manual > JSON
    if args.w is not None and args.T_ram is not None and args.T_dense is not None:
        w = float(args.w)
        T_ram = max(float(args.T_ram), 1e-3)
        T_dense = max(float(args.T_dense), 1e-3)
        print(f"[Manual fusion params] T_RAM={T_ram:.3f} | T_Dense={T_dense:.3f} | w={w:.2f}")
    elif args.fusion_params is not None:
        with open(args.fusion_params, "r") as f:
            params = json.load(f)
        T_ram   = max(float(params["T_ram"]), 1e-3)
        T_dense = max(float(params["T_dense"]), 1e-3)
        w       = float(params["w"])
        print(f"[Loaded fusion params] T_RAM={T_ram:.3f} | T_Dense={T_dense:.3f} | w={w:.2f}")
    else:
        raise ValueError("Provide manual params (--w --T_ram --T_dense) OR --fusion_params JSON.")

    ram_temp = TemperatureScaler(T_ram).to(device);   ram_temp.log_T.data = torch.log(torch.tensor(T_ram))
    dn_temp  = TemperatureScaler(T_dense).to(device); dn_temp.log_T.data  = torch.log(torch.tensor(T_dense))

    # === Build TEST loader(s)
    test_df = pd.read_csv(args.test_csv)
    if not {"Image_path", "label"}.issubset(test_df.columns):
        raise ValueError("--test_csv must have columns: Image_path,label")
    expand = not args.no_tta
    ram_test_ds = TTADatasetPerModel(test_df, args.test_image_root, ram_transform,   expand_tta=expand)
    dn_test_ds  = TTADatasetPerModel(test_df, args.test_image_root, dense_transform, expand_tta=expand)
    ram_loader = DataLoader(ram_test_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    dn_loader  = DataLoader(dn_test_ds,  batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # === Evaluate on TEST
    all_logits, all_labels = [], []
    dn_iter = iter(dn_loader)
    with torch.no_grad():
        for x_ram, y in tqdm(ram_loader, desc="TEST (manual fusion)"):
            z_ram = logits_with_tta(ram_model, x_ram, device, temp_scaler=ram_temp)
            x_dn, _ = next(dn_iter)
            z_dn  = logits_with_tta(dense_model, x_dn, device, temp_scaler=dn_temp)
            z_fuse = (1 - w) * z_ram + w * z_dn
            all_logits.append(z_fuse)
            all_labels.append(y.to(device))

    z_test = torch.cat(all_logits, 0)
    y_test = torch.cat(all_labels, 0)

    acc = (z_test.argmax(1) == y_test).float().mean().item()
    cm = confusion_matrix(z_test, y_test, num_classes=4).cpu().numpy()
    print(f"\n[TEST FUSION manual] Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)

    if args.pred_csv is not None:
        preds = z_test.argmax(1).cpu().numpy()
        out = test_df.copy()
        out["pred_label"] = preds + 1  # back to 1..4
        out.to_csv(args.pred_csv, index=False)
        print(f"[Saved] predictions → {args.pred_csv}")

if __name__ == "__main__":
    main()
