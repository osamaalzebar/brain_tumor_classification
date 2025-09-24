import os
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax

from torchvision.models import densenet201, DenseNet201_Weights

# --- import your dataset utilities (same file you used for training) ---
from dataset_densenet import build_transforms, INV_LABEL_MAP, LABEL_MAP


# ----------------------------- utils -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------- multi-branch DenseNet ---------------------
class MultiBranchDenseNet201(nn.Module):
    """
    Must match the training-time architecture exactly.
    Taps Dense blocks {2, 3, 4}, each head: GAP -> Dropout -> FC(->256) -> ReLU,
    then concat (3*256) -> final classifier.
    """
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

        def make_head(in_ch):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Dropout(p=dropout_p),
                nn.Linear(in_ch, branch_dim),
                nn.ReLU(inplace=True),
            )

        self.head2 = make_head(ch_db2)
        self.head3 = make_head(ch_db3)
        self.head4 = make_head(ch_db4)

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
        z  = torch.cat([z2, z3, z4], dim=1)
        return self.classifier(z)


# ----------------------------- dataset ----------------------------
class MRITestDataset(Dataset):
    """
    Reads CSV with columns:
      - required: 'Image_path'
      - optional: 'label' (in {1,2,3,4})
    Returns (image_tensor, label_or_None, image_name).
    """
    def __init__(self, image_dir: str, csv_file: str, transform, strict_exists: bool = True):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        if "Image_path" not in self.df.columns:
            raise ValueError("CSV must have 'Image_path' column")
        self.has_label = "label" in self.df.columns
        self.transform = transform
        self.strict_exists = strict_exists

        self.samples: List[Tuple[str, Optional[int], str]] = []
        for _, row in self.df.iterrows():
            img_name = str(row["Image_path"])
            full_path = os.path.join(self.image_dir, img_name)
            if strict_exists and not os.path.exists(full_path):
                raise FileNotFoundError(f"Image not found: {full_path}")

            y = None
            if self.has_label:
                y_raw = int(row["label"])
                if y_raw not in LABEL_MAP:
                    raise ValueError(f"Unexpected label {y_raw} for {img_name}")
                y = LABEL_MAP[y_raw]  # map to 0..C-1

            self.samples.append((full_path, y, img_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y, img_name = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y_tensor = None if y is None else torch.tensor(y, dtype=torch.long)
        return img, y_tensor, img_name


# ----------------------------- test -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Folder containing images referenced in CSV")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV with Image_path and optional label")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to best_model.pt saved by training")
    parser.add_argument("--out_csv", type=str, default="predictions.csv",
                        help="Where to write predictions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=None,
                        help="If None, will use img_size stored in checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    num_classes = int(ckpt.get("num_classes", 4))
    ckpt_img_size = int(ckpt.get("img_size", 224))
    img_size = int(args.img_size) if args.img_size is not None else ckpt_img_size

    # --- Build model & load weights ---
    model = MultiBranchDenseNet201(num_classes=num_classes, branch_dim=256,
                                   dropout_p=0.5, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    # --- Data ---
    transform = build_transforms(img_size=img_size, train=False)
    ds = MRITestDataset(args.image_dir, args.csv_file, transform, strict_exists=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # --- Inference ---
    rows = []
    has_label = ds.has_label
    correct, total = 0, 0

    with torch.no_grad():
        autocast_on = torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=autocast_on):
            for imgs, y, names in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = softmax(logits, dim=1)
                confs, preds = probs.max(dim=1)

                preds_np = preds.cpu().numpy()
                confs_np = confs.cpu().numpy()

                if has_label:
                    y_np = y.cpu().numpy()
                    correct += (preds_np == y_np).sum()
                    total += y_np.shape[0]

                for i in range(len(names)):
                    pred_idx = int(preds_np[i])                  # 0..C-1
                    pred_label = int(INV_LABEL_MAP[pred_idx])    # map back to {1,2,3,4}
                    row = {
                        "Image_path": names[i],
                        "pred_index": pred_idx,
                        "pred_label": pred_label,
                        "pred_prob": float(confs_np[i]),
                    }
                    if has_label:
                        true_idx = int(y_np[i])
                        true_label = int(INV_LABEL_MAP[true_idx])
                        row["true_label"] = true_label
                        row["correct"] = int(pred_idx == true_idx)
                    rows.append(row)

    # --- Save predictions ---
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote predictions to: {args.out_csv}")

    # --- Metrics if labels exist ---
    if has_label and total > 0:
        acc = correct / total
        print(f"Accuracy: {acc:.4f}")

        try:
            from sklearn.metrics import confusion_matrix
            true_labels = out_df["true_label"].to_numpy()
            pred_labels = out_df["pred_label"].to_numpy()
            cm = confusion_matrix(true_labels, pred_labels, labels=[1,2,3,4])
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm)
        except Exception as e:
            print(f"(Skipping confusion matrix: {e})")


if __name__ == "__main__":
    main()
