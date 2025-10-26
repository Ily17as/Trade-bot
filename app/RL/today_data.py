# make_today_data.py
# Создаёт:
#   1) FEAT_TODAY = табличные фичи за сегодня
#   2) CV_MANIFEST = манифест PNG за сегодня (и сами PNG)
#
# Зависимости: pandas, numpy, pyarrow, scripts.render_cv_images

import os
from pathlib import Path
import pandas as pd
import numpy as np

# --- ПАРАМЕТРЫ ---
TICKER = os.getenv("TICKER", "SBER")
TF     = os.getenv("TF", "5m")      # "5m"/"15m"/"1h"
H      = int(os.getenv("H", "36"))  # для 5m и 180 мин → 36

# Источник всех фич (история)
FEAT_SRC   = os.getenv("FEAT_SRC",  f"../../data/features/ml/{TICKER}/{TF}/features_labeled.parquet")
# Куда положить «сегодня»
FEAT_TODAY = os.getenv("FEAT_TODAY",f"../../data/features/ml/{TICKER}/{TF}/features_today.parquet")
# Куда положить манифест картинок «сегодня»
CV_MANIFEST= os.getenv("CV_MANIFEST",f"../../data/cv/images/{TICKER}/{TF}/today_manifest.csv")

# Рендер картинок (совместимо с твоим scripts.render_cv_images)
RENDER_SIZE = 64
RENDER_STEP = 16
VOL_MODE    = "zscore"
FIXED_YLIM  = "window"
YLIM_PAD    = 0.02
OHLC_MODE   = "close+hl_wick"
EXTRA_CH    = "atr_rel"   # опционально, если есть такая колонка
SEED        = 42

def ensure_dir(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def load_today(feat_src: str) -> pd.DataFrame:
    df = pd.read_parquet(feat_src).sort_values("time").reset_index(drop=True)
    ts = pd.to_datetime(df["time"], utc=True)

    # было: pd.Timestamp.utcnow().normalize().tz_localize("UTC")  # падало
    start = pd.Timestamp.now(tz="UTC").normalize()        # начало сегодняшних суток (UTC)
    end   = start + pd.Timedelta(days=1)

    mask = (ts >= start) & (ts < end)
    df_today = df.loc[mask].copy()
    if df_today.empty:
        df_today = df.tail(500).copy()
        print("[warn] За сегодня пусто. Взял последние 500 строк истории как proxy.")
    return df_today

def save_today(df_today: pd.DataFrame, path_out: str):
    ensure_dir(path_out)
    df_today.to_parquet(path_out, index=False)
    print(f"[ok] FEAT_TODAY -> {path_out} rows={len(df_today)} "
          f"{df_today['time'].iloc[0]} → {df_today['time'].iloc[-1]}")

def render_today(in_parquet: str, manifest_csv: str):
    from scripts.render_cv_images import render_with_manifest

    # если нет строк — не рендерим
    df = pd.read_parquet(in_parquet)
    if len(df) < RENDER_SIZE:
        print("[warn] Слишком мало строк для рендера окна. Пропуск.")
        return

    ensure_dir(manifest_csv)
    rows = render_with_manifest(
        in_parquet = in_parquet,
        ticker     = TICKER,
        tf         = TF,
        size       = RENDER_SIZE,
        step       = RENDER_STEP,
        horizon    = H,
        show_axes  = False,
        show_volume= True,
        vol_mode   = VOL_MODE,
        fixed_ylim = FIXED_YLIM,
        ylim_pad   = YLIM_PAD,
        ohlc_mode  = OHLC_MODE,
        extra_ch   = EXTRA_CH if EXTRA_CH in df.columns else None,
        manifest_csv = manifest_csv,
        seed = SEED
    )
    print(f"[ok] CV_MANIFEST -> {manifest_csv} rows={len(rows)}")

if __name__ == "__main__":
    # 1) FEAT_TODAY
    if not Path(FEAT_SRC).exists():
        raise FileNotFoundError(f"FEAT_SRC not found: {FEAT_SRC}")
    df_today = load_today(FEAT_SRC)
    # базовая очистка числовых фич (без утечек): inf→NaN→ffill→0
    num_cols = df_today.select_dtypes(include=[np.number, "bool"]).columns
    for c in df_today.select_dtypes(include=["bool"]).columns:
        df_today[c] = df_today[c].astype(np.uint8)
    df_today[num_cols] = (df_today[num_cols]
                          .replace([np.inf, -np.inf], np.nan)
                          .ffill().fillna(0.0))
    save_today(df_today, FEAT_TODAY)

    # 2) CV_MANIFEST + PNG
    render_today(FEAT_TODAY, CV_MANIFEST)
