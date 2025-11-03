# baseline_cpu_nothreads.py
# requirements: timm, torch, torchvision, tqdm, pandas, pillow
import os
import json
import time
from pathlib import Path

import torch
import timm
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

# ----------------- CONFIG -----------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
LABELS_CSV = "../../data/labels.csv"      # adjust if needed
IMAGES_ROOT = "../../cv_candles_seq/images"
MODEL_NAME = "convnext_tiny"              # light model for CPU
PRETRAINED = True                         # set False to avoid downloading weights
BATCH_SIZE = 16                           # smaller for CPU
VAL_BATCH = 64
NUM_EPOCHS = 10
DEVICE = torch.device("cpu")
# ------------------------------------------

print("PyTorch device:", DEVICE)
print("Using timm model:", MODEL_NAME, "pretrained=", PRETRAINED)

# ---- load labels csv and build label_map ----
df = pd.read_csv(LABELS_CSV)
df['label'] = df['label'].astype(str).str.strip().str.lower()

print("Unique labels in CSV:", sorted(df['label'].unique().tolist()))
unique_labels = sorted(df['label'].unique())
label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
df['label_idx'] = df['label'].map(label_to_idx).astype(int)
df['filename'] = df['filename'].apply(lambda x: os.path.join(IMAGES_ROOT, x))

# quick sanity checks
missing = (~df['filename'].apply(os.path.exists)).sum()
if missing:
    print(f"Warning: {missing} image paths not found. First missing (if any):")
    print(df.loc[~df['filename'].apply(os.path.exists), 'filename'].head(3))
else:
    print("All image paths exist (quick check).")

# save meta
meta = {"label_to_idx": label_to_idx, "input_size": [3, 224, 224], "model_name": MODEL_NAME}
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ---- Dataset ----
class ImgDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        p = self.df.loc[i, 'filename']
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        y = int(self.df.loc[i, 'label_idx'])
        return x, torch.tensor(y, dtype=torch.long)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

# num_workers=0 -> no subprocesses for loading (safe on Windows / CPU-only)
train_dl = DataLoader(ImgDataset(train_df, transform),
                      batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_dl   = DataLoader(ImgDataset(val_df, transform),
                      batch_size=VAL_BATCH, shuffle=False, num_workers=0, pin_memory=False)

# --- create backbone (diagnostics) ---
print("Creating model. If pretrained=True this may download weights to cache (~/.cache/torch/...).")
t0 = time.time()
try:
    backbone = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=0, global_pool='avg')
except Exception as e:
    print("Model creation failed:", e)
    print("Retrying with pretrained=False ...")
    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, global_pool='avg')
t1 = time.time()
print(f"Model created in {t1-t0:.1f}s. num_features={backbone.num_features}")

feat_dim = backbone.num_features
for p in backbone.parameters():
    p.requires_grad = False

head = nn.Sequential(
    nn.Linear(feat_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(label_to_idx))
)

model = nn.Sequential(backbone, head).to(DEVICE)

# optimizer and loss
opt = optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-4)
class_weights = torch.ones(len(label_to_idx), dtype=torch.float32).to(DEVICE)
crit = nn.CrossEntropyLoss(weight=class_weights)

# training loop with tqdm and proper loss.item()
best_val = -1.0
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    n_samples = 0
    pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for xb, yb in pbar:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        batch_loss = loss.item()
        train_loss += batch_loss * xb.size(0)
        n_samples += xb.size(0)
        pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

    avg_train_loss = train_loss / max(1, n_samples)

    # validation
    model.eval()
    tot, ok = 0, 0
    with torch.no_grad():
        for xb, yb in tqdm(val_dl, desc=f"Epoch {epoch:02d} [val]  ", leave=False):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            tot += yb.size(0)
            ok += (preds == yb).sum().item()

    val_acc = ok / tot if tot > 0 else 0.0
    print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}  val_acc={val_acc:.4f}")

    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "val_acc": val_acc,
        "label_to_idx": label_to_idx
    }
    torch.save(ckpt, MODEL_DIR / f"chk_epoch{epoch:03d}.pth")
    if val_acc > best_val:
        best_val = val_acc
        torch.save(ckpt, MODEL_DIR / "best_model.pth")

print("Training finished. Best val_acc:", best_val)
