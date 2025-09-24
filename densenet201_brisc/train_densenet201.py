import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import densenet201, DenseNet201_Weights

from mri_dataset import MRIDataset, build_transforms


# ----------------------------- utils -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------- multi-branch DenseNet ---------------------
class MultiBranchDenseNet201(nn.Module):
    """
    Matches the diagram:
      taps from Dense blocks {2, 3, 4} (i.e., 12/48/32 conv blocks)
      each tap: GAP -> Dropout -> FC -> ReLU
      concat three heads -> final classifier
    """
    def __init__(self, num_classes=4, branch_dim=256, dropout_p=0.5, pretrained=True):
        super().__init__()
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        base = densenet201(weights=weights)

        # --- DenseNet feature extractor pieces (as in torchvision) ---
        feats = base.features
        self.stem = nn.Sequential(feats.conv0, feats.norm0, feats.relu0, feats.pool0)
        self.db1 = feats.denseblock1; self.tr1 = feats.transition1
        self.db2 = feats.denseblock2; self.tr2 = feats.transition2
        self.db3 = feats.denseblock3; self.tr3 = feats.transition3
        self.db4 = feats.denseblock4; self.out_norm = feats.norm5

        # Channels *before* transitions for blocks 2/3, and after norm for block 4
        ch_db2 = 512    # after db2, before tr2
        ch_db3 = 1792   # after db3, before tr3
        ch_db4 = 1920   # after db4 + norm5

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
        # stem + block1
        x = self.stem(x)
        x = self.db1(x); x = self.tr1(x)

        # block2 (tap BEFORE transition2)
        x = self.db2(x)
        tap2 = x
        x = self.tr2(x)

        # block3 (tap BEFORE transition3)
        x = self.db3(x)
        tap3 = x
        x = self.tr3(x)

        # block4 (final + norm)
        x = self.db4(x)
        tap4 = self.out_norm(x)

        # heads
        z2 = self.head2(tap2)
        z3 = self.head3(tap3)
        z4 = self.head4(tap4)
        z = torch.cat([z2, z3, z4], dim=1)

        return self.classifier(z)


# ------------------------------ train -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_classes", type=int, default=4)  # <— 4 as requested
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # splits
    df = pd.read_csv(args.csv_file)
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=args.seed, stratify=df["label"]
    )
    train_csv = os.path.join(args.out_dir, "train_split.csv")
    val_csv = os.path.join(args.out_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # data
    train_ds = MRIDataset(args.image_dir, train_csv, build_transforms(args.img_size, train=True))
    val_ds  = MRIDataset(args.image_dir, val_csv,  build_transforms(args.img_size, train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # model (matches diagram)
    model = MultiBranchDenseNet201(num_classes=args.num_classes, branch_dim=256,
                                   dropout_p=0.5, pretrained=True)

    if args.freeze_backbone:
        for m in [model.stem, model.db1, model.tr1, model.db2, model.tr2,
                  model.db3, model.tr3, model.db4, model.out_norm]:
            for p in m.parameters():
                p.requires_grad = False
        # heads + classifier remain trainable

    model = model.to(device)

    # opt
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # train
    best_val_acc = 0.0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_acc, n = 0.0, 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = labels.size(0)
            train_loss += loss.item() * bs
            train_acc += accuracy(logits, labels) * bs
            n += bs
        train_loss /= n
        train_acc  /= n

        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc  += accuracy(logits, labels) * bs
                m += bs
        val_loss /= m
        val_acc  /= m
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "img_size": args.img_size,
                "model": "MultiBranchDenseNet201",
                "num_classes": args.num_classes,
            }, best_path)
            print(f"  ↳ Saved new best to {best_path} (val_acc={best_val_acc:.4f})")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
