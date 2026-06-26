import os
import math
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# ── CONFIG ──────────────────────────────────────────────
TRAIN_CSV   = "archive(2)/train_1.csv"
TRAIN_DIR   = "archive(2)/train_images/train_images"
VAL_CSV     = "archive(2)/valid.csv"
VAL_DIR     = "archive(2)/val_images"
MODEL_DIR   = "model"
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 12        # ↑ from 6
PATIENCE    = 3         # early stopping patience
LR_RESNET   = 1e-4      # ResNet learning rate
LR_EFFNET   = 5e-5      # EfficientNet needs lower LR to converge well
WARMUP_EPOCHS = 2       # linear LR warmup for first N epochs
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  EyeCheck AI — Improved Dual Model Trainer")
print(f"  Device  : {DEVICE}")
print(f"  Epochs  : {EPOCHS} (early stop patience={PATIENCE})")
print(f"  LR      : ResNet={LR_RESNET}  EfficientNet={LR_EFFNET}")
print(f"  Schedule: {WARMUP_EPOCHS}-epoch warmup → cosine decay")
print(f"  Imbalance: WeightedSampler + Weighted Loss")
print(f"{'='*60}\n")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── LABEL MAPPING ────────────────────────────────────────
def map_label(x):
    if x == 0: return 0
    if x == 1: return 1
    return 2


# ── DATASET ──────────────────────────────────────────────
class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = os.path.normpath(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = row.get("id_code", row.get("image", str(row.iloc[0])))
        path  = os.path.join(self.img_dir, str(fname) + ".png")
        if not os.path.exists(path):
            path = os.path.join(self.img_dir, str(fname) + ".jpg")
        image = Image.open(path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ── TRANSFORMS ───────────────────────────────────────────
# Stronger augmentation to improve generalization
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # slightly larger then crop
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


# ── CLASS WEIGHTS ────────────────────────────────────────
def compute_class_weights(labels, num_classes=3):
    counts  = Counter(labels)
    total   = len(labels)
    weights = []
    for c in range(num_classes):
        freq = counts.get(c, 1) / total
        weights.append(1.0 / freq)
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes
    print(f"  Class weights : {[f'{w:.3f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

def make_sampler(labels):
    counts        = Counter(labels)
    class_weight  = {c: 1.0 / counts[c] for c in counts}
    sample_weights = [class_weight[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ── LR SCHEDULER WITH WARMUP ─────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """
    Linear warmup for first warmup_epochs, then cosine decay to 0.
    """
    total_steps  = total_epochs  * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── LOAD DATA ────────────────────────────────────────────
def load_dataframes():
    def prep(df):
        col = "diagnosis" if "diagnosis" in df.columns else df.columns[-1]
        df["label"] = df[col].apply(map_label)
        return df

    train_df = prep(pd.read_csv(TRAIN_CSV))

    if os.path.exists(VAL_CSV):
        val_df = prep(pd.read_csv(VAL_CSV))
        val_dir = VAL_DIR
        print(f"Using pre-split val set from {VAL_CSV}")
    else:
        train_df, val_df = train_test_split(
            train_df, test_size=0.15,
            stratify=train_df["label"], random_state=42
        )
        val_dir = TRAIN_DIR
        print("No val CSV — splitting 85/15 from train")

    print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")
    print(f"Class distribution (train):")
    for c, name in enumerate(["No DR", "Mild DR", "Moderate+"]):
        cnt = (train_df["label"] == c).sum()
        pct = cnt / len(train_df) * 100
        bar = "█" * int(pct / 3)
        print(f"  {name:12s} [{c}]: {cnt:4d} ({pct:4.1f}%) {bar}")
    print()
    return train_df, val_df, val_dir

train_df, val_df, val_dir = load_dataframes()

train_labels  = list(train_df["label"])
class_weights = compute_class_weights(train_labels)
sampler       = make_sampler(train_labels)

# Check val images location
if not os.path.exists(val_dir):
    val_dir = TRAIN_DIR
    print(f"Val dir not found, using train dir for val images")

# Also check for nested val folder
nested_val = os.path.join(val_dir, "val_images")
if os.path.exists(nested_val):
    val_dir = nested_val
    print(f"Using nested val dir: {val_dir}")

train_ds = APTOSDataset(train_df, TRAIN_DIR,  train_tf)
val_ds   = APTOSDataset(val_df,   val_dir,    val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)


# ── TRAIN FUNCTION ───────────────────────────────────────
def train_one_model(model, model_name, save_path, lr):
    print(f"\n{'─'*60}")
    print(f"  Training : {model_name}  |  LR={lr}")
    print(f"{'─'*60}")

    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = get_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, len(train_loader))

    best_acc       = 0.0
    patience_count = 0

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += imgs.size(0)

            if (i + 1) % 30 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  [{model_name}] Ep {epoch+1}/{EPOCHS} "
                      f"step {i+1}/{len(train_loader)} "
                      f"loss={loss.item():.4f}  lr={current_lr:.2e}")

        # ── Validate ──
        model.eval()
        v_correct, v_total = 0, 0
        per_class_correct  = [0, 0, 0]
        per_class_total    = [0, 0, 0]

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total   += imgs.size(0)
                for c in range(3):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c]   += mask.sum().item()

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total

        print(f"\n  Epoch {epoch+1}/{EPOCHS} → "
              f"Train Acc: {t_acc:.3f} | Val Acc: {v_acc:.3f}")
        for c, name in enumerate(["No DR", "Mild", "Moderate+"]):
            if per_class_total[c] > 0:
                ca = per_class_correct[c] / per_class_total[c]
                bar = "█" * int(ca * 20)
                print(f"    {name:10s}: {ca:.3f}  {bar}")

        # ── Save best ──
        if v_acc > best_acc:
            best_acc       = v_acc
            patience_count = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Saved {model_name} (Val Acc: {v_acc:.3f})\n")
        else:
            patience_count += 1
            print(f"  ⏳ No improvement ({patience_count}/{PATIENCE})\n")
            if patience_count >= PATIENCE:
                print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n  {model_name} complete. Best Val Acc: {best_acc:.3f}")
    return best_acc


# ════════════════════════════════════════════════════════
#  MODEL 1 — ResNet50
# ════════════════════════════════════════════════════════
print("\nBuilding ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Unfreeze layer3 + layer4 + fc for more capacity
for name, p in resnet.named_parameters():
    if not any(x in name for x in ["layer3", "layer4", "fc"]):
        p.requires_grad = False

resnet.fc = nn.Sequential(
    nn.BatchNorm1d(resnet.fc.in_features),
    nn.Dropout(0.4),
    nn.Linear(resnet.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 3)
)

trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

acc_r = train_one_model(
    resnet, "ResNet50",
    os.path.join(MODEL_DIR, "resnet50_dr.pth"),
    lr=LR_RESNET
)


# ════════════════════════════════════════════════════════
#  MODEL 2 — EfficientNet-B4
# ════════════════════════════════════════════════════════
print("\nBuilding EfficientNet-B4...")
effnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

# Unfreeze last 3 feature blocks (was 2) + classifier
blocks = list(effnet.features.children())
for blk in blocks[:-3]:            # freeze all except last 3 blocks
    for p in blk.parameters():
        p.requires_grad = False

in_f = effnet.classifier[1].in_features
effnet.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_f, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 3)
)

trainable = sum(p.numel() for p in effnet.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")

acc_e = train_one_model(
    effnet, "EfficientNet-B4",
    os.path.join(MODEL_DIR, "efficientnet_dr.pth"),
    lr=LR_EFFNET
)


# ── FINAL SUMMARY ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ✅ Training Complete!")
print(f"{'─'*60}")
print(f"  ResNet50       Best Val Acc : {acc_r:.3f}  ({acc_r*100:.1f}%)")
print(f"  EfficientNet   Best Val Acc : {acc_e:.3f}  ({acc_e*100:.1f}%)")
ensemble_est = (acc_r + acc_e) / 2
print(f"  Ensemble est.  ~Val Acc     : {ensemble_est:.3f}  ({ensemble_est*100:.1f}%)")
print(f"{'─'*60}")
print(f"  Saved:")
print(f"  → model/resnet50_dr.pth")
print(f"  → model/efficientnet_dr.pth")
print(f"\n  Now run:  streamlit run app.py")
print(f"{'='*60}\n")