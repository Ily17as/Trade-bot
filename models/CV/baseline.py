import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import joblib

# ---------- настройки ----------
MANIFEST = r"../data/cv/images/SBER/5m/batch_0.csv"  # путь к манифесту
PROJECT_ROOT = os.path.abspath("../..")                 # корень проекта для относительных путей
SEED = 42
EPOCHS = 8
BATCH = 256

# ---------- загрузка и починка манифеста ----------
LABELS_MAP = {"down": 0, "flat": 1, "up": 2, -1: 0, 0: 1, 1: 2}

def load_manifest(path_csv: str, root: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    if "label" not in df.columns or "path" not in df.columns:
        raise KeyError("В манифесте должны быть колонки 'path' и 'label'.")

    # 1) нормализуем метки: принимаем строки и {-1,0,1}
    lab = pd.to_numeric(df["label"], errors="coerce")
    if lab.notna().any():
        df["label"] = lab.map(LABELS_MAP)
    else:
        s = df["label"].astype(str).str.lower().str.strip()
        s = s.replace({"none": "", "nan": "", "": np.nan})
        df["label"] = s.map(LABELS_MAP)

    # 2) чиним пути: если файла нет, пробуем прибавить root
    def fix_path(p):
        if not isinstance(p, str):
            return None
        p_norm = os.path.normpath(p)
        if os.path.isfile(p_norm):
            return p_norm
        if root:
            q = os.path.normpath(os.path.join(root, p_norm))
            return q if os.path.isfile(q) else None
        return None

    df["path"] = df["path"].map(fix_path)

    # 3) фильтры
    keep = df["label"].notna() & df["path"].notna()
    df = df[keep].copy()

    # временной сплит по t_end если есть
    if "t_end" in df.columns:
        df["t_end"] = pd.to_datetime(df["t_end"], errors="coerce")
        df = df.dropna(subset=["t_end"]).sort_values("t_end")

    if len(df) == 0:
        raise ValueError("После нормализации пусто: проверь значения 'label' и столбец 'path'.")

    df["label"] = df["label"].astype(int)
    return df

df = load_manifest(MANIFEST, root=PROJECT_ROOT)
print("rows:", len(df))
print("label dist:", df["label"].value_counts().to_dict())
print("exists %:", df["path"].map(os.path.isfile).mean())

# ---------- сплиты ----------
rng = np.random.RandomState(SEED)
if "t_end" in df.columns:
    cut = df["t_end"].max() - pd.Timedelta(days=30)
    tr_df = df[df["t_end"] < cut]
    va_df = df[df["t_end"] >= cut]
    if len(tr_df) == 0 or len(va_df) == 0:
        msk = rng.rand(len(df)) < 0.8
        tr_df, va_df = df[msk], df[~msk]
else:
    msk = rng.rand(len(df)) < 0.8
    tr_df, va_df = df[msk], df[~msk]

assert len(tr_df) > 0 and len(va_df) > 0, "Нужно >0 объектов в train и val"
print(f"train={len(tr_df)}  val={len(va_df)}")

# ---------- датасет ----------
class ImgDS(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.df = frame.reset_index(drop=True)
        self.t = tv.transforms.Compose([
            tv.transforms.Grayscale(num_output_channels=1),
            tv.transforms.ToTensor(),                 # -> [1, 64, 64]
            tv.transforms.Normalize([0.5], [0.5]),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        p = self.df.path[i]
        x = self.t(Image.open(p).convert("L"))
        y = int(self.df.label[i])
        return x, y

tr_ds, va_ds = ImgDS(tr_df), ImgDS(va_df)

# ---------- лоадеры (Windows-safe) ----------
num_workers = 0
batch = min(BATCH, max(8, len(tr_ds)//4))
tr = DataLoader(tr_ds, batch_size=batch, shuffle=True,  num_workers=num_workers, pin_memory=True)
va = DataLoader(va_ds, batch_size=min(batch*2, len(va_ds)), shuffle=False, num_workers=num_workers, pin_memory=True)

# ---------- модель ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = tv.models.resnet18(weights=None, num_classes=3)
m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
m.to(device)

# class weights от дисбаланса
vals = tr_df["label"].value_counts(normalize=True).to_dict()
w = torch.tensor([1/vals.get(0,1e-9), 1/vals.get(1,1e-9), 1/vals.get(2,1e-9)], dtype=torch.float32, device=device)

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(weight=w)

# ---------- обучение ----------
for epoch in range(EPOCHS):
    m.train()
    loss_sum = 0.0
    for xb, yb in tr:
        xb = xb.to(device); yb = torch.as_tensor(yb, device=device)
        opt.zero_grad()
        out = m(xb)
        loss = crit(out, yb)
        loss.backward(); opt.step()
        loss_sum += loss.item() * xb.size(0)

    m.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for xb, yb in va:
            xb = xb.to(device); yb = torch.as_tensor(yb, device=device)
            pred = m(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    print(f"epoch {epoch:02d} | train_loss={loss_sum/max(1,len(tr_ds)):.4f} | val_acc={correct/max(1,total):.3f}")

# joblib.dump(m.cpu(), "../models/cv_resnet18.pkl")
torch.save(m.cpu().state_dict(), "../models/cv_resnet18.pt")