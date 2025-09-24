import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from dataset_RAM import BrainMRIDataset, get_transforms
from ram.ram.models import ram

# === Config ===
# Train (manually split)
train_csv_path  = '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/train/image_labels.csv'
train_image_root = '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/train/data'

# Validation (manually split)
val_csv_path  = '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/val/image_labels.csv'  # <- as provided
val_image_root = '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/fuse/classification_task/val/data'

ram_pretrained = '/mnt/d9b25a57-72aa-4635-b6e9-bf7337153bb8/brain_tumor/brain_tumor_pyramid_brisc_augment_no_5_fold/checkpoint/ram_swin_large_14m.pth'

checkpoint_path = "checkpoints/ram_finetuned_brain_tumor_best.pth"
num_epochs = 30
batch_size = 4
lr = 5e-5
weight_decay = 1e-4
num_workers = 4
image_size = 384
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RAM wrapper ===
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

        x = self.ram.visual_encoder.layers[0](x)          # Stage 1

        x2 = self.ram.visual_encoder.layers[1](x)          # Stage 2
        feat2 = x2.permute(0, 2, 1).reshape(B, -1, 24, 24)
        feat2 = self.pool(feat2).squeeze(-1).squeeze(-1)
        feat2 = self.branch2(feat2)

        x3 = self.ram.visual_encoder.layers[2](x2)         # Stage 3
        feat3 = x3.permute(0, 2, 1).reshape(B, -1, 12, 12)
        feat3 = self.pool(feat3).squeeze(-1).squeeze(-1)
        feat3 = self.branch3(feat3)

        x4 = self.ram.visual_encoder.layers[3](x3)         # Stage 4
        feat4 = x4.permute(0, 2, 1).reshape(B, -1, 6, 6)
        feat4 = self.pool(feat4).squeeze(-1).squeeze(-1)
        feat4 = self.branch4(feat4)

        concat_feat = torch.cat([feat2, feat3, feat4], dim=1)
        logits = self.classifier(concat_feat)
        return logits

def main():
    # === Load pre-split dataframes ===
    train_df = pd.read_csv(train_csv_path)
    val_df   = pd.read_csv(val_csv_path)

    # === Datasets & loaders (augment on train only) ===
    train_dataset = BrainMRIDataset(train_df, train_image_root, transform=get_transforms(augment=True))
    val_dataset   = BrainMRIDataset(val_df,   val_image_root,   transform=get_transforms(augment=False))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # === Load RAM backbone ===
    base_model = ram(
        pretrained=ram_pretrained,
        vit='swin_l',
        image_size=image_size
    ).to(device)

    # === Freeze & (optionally) unfreeze deeper layers ===
    for param in base_model.visual_encoder.parameters():
        param.requires_grad = False
    for name, param in base_model.visual_encoder.named_parameters():
        if 'layers.1' in name or 'layers.2' in name or 'layers.3' in name:
            param.requires_grad = True

    model = RAMWithClassifier(base_model, num_classes=num_classes).to(device)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        # === Train ===
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, labels_batch = images.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_acc = correct / total if total else 0.0
        avg_train_loss = running_loss / max(len(train_loader), 1)

        # === Validate ===
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_acc = val_correct / val_total if val_total else 0.0

        print(f"Epoch {epoch+1:02}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # === Save best ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best model to {checkpoint_path} (Val Acc: {best_val_acc:.4f})")

    print(f"\nTraining done. Best Val Accuracy: {best_val_acc:.4f} (checkpoint: {checkpoint_path})")

if __name__ == "__main__":
    main()
