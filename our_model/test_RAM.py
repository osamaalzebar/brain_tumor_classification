import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

from dataset_RAM import BrainMRIDataset, get_transforms
from ram.ram.models import ram

# === Paths (update if needed) ===
TEST_CSV = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/image_labels_test.csv"
TEST_IMG_ROOT = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/test/data"

# Use the same pretrained path you used in training:
RAM_PRETRAINED = "/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/checkpoint/ram_swin_large_14m.pth"

# Best checkpoint produced by the new training script
CHECKPOINT_PATH = "checkpoints/ram_finetuned_brain_tumor_best.pth"

BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_CLASSES = 4
IMAGE_SIZE = 384

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Must match the one used in training ===
class RAMWithClassifier(nn.Module):
    def __init__(self, ram_model, embed_dims=[768, 1536, 6144], dropout=0.3, num_classes=4):
        super().__init__()
        self.ram = ram_model

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.branch2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dims[0], 256),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dims[1], 256),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dims[2], 256),
            nn.ReLU()
        )

        self.classifier = nn.Linear(256 * 3, num_classes)

    def forward(self, image):
        B = image.size(0)
        x = self.ram.visual_encoder.patch_embed(image)
        x = self.ram.visual_encoder.pos_drop(x)

        # Stage 1 (skipped)
        x = self.ram.visual_encoder.layers[0](x)

        # Stage 2
        x2 = self.ram.visual_encoder.layers[1](x)
        feat2 = x2.permute(0, 2, 1).reshape(B, -1, 24, 24)
        feat2 = self.pool(feat2).squeeze(-1).squeeze(-1)
        feat2 = self.branch2(feat2)

        # Stage 3
        x3 = self.ram.visual_encoder.layers[2](x2)
        feat3 = x3.permute(0, 2, 1).reshape(B, -1, 12, 12)
        feat3 = self.pool(feat3).squeeze(-1).squeeze(-1)
        feat3 = self.branch3(feat3)

        # Stage 4
        x4 = self.ram.visual_encoder.layers[3](x3)
        feat4 = x4.permute(0, 2, 1).reshape(B, -1, 6, 6)
        feat4 = self.pool(feat4).squeeze(-1).squeeze(-1)
        feat4 = self.branch4(feat4)

        concat_feat = torch.cat([feat2, feat3, feat4], dim=1)
        logits = self.classifier(concat_feat)
        return logits


def load_base_ram():
    base_model = ram(
        pretrained=RAM_PRETRAINED,
        vit='swin_l',
        image_size=IMAGE_SIZE
    ).to(device)
    # Freeze backbone params (same as training before classifier)
    for p in base_model.visual_encoder.parameters():
        p.requires_grad = False
    return base_model


def evaluate_checkpoint(ckpt_path, test_loader):
    base_model = load_base_ram()
    model = RAMWithClassifier(base_model, num_classes=NUM_CLASSES).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {os.path.basename(ckpt_path)}"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, prec, rec


def main():
    # Test dataset/loader (no augmentation)
    df_test = pd.read_csv(TEST_CSV)
    test_dataset = BrainMRIDataset(df_test, TEST_IMG_ROOT, transform=get_transforms(augment=False))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    acc, prec, rec = evaluate_checkpoint(CHECKPOINT_PATH, test_loader)
    print("\n==================== Test Results ====================")
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision (macro)  : {prec:.4f}")
    print(f"Recall    (macro)  : {rec:.4f}")


if __name__ == "__main__":
    main()
