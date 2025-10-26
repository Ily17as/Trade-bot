# backfill_signals.py — per-bar backfill for RL signals with CV (path-fix)
# deps: pandas numpy pyarrow joblib torch torchvision pillow
import os, math, numpy as np, pandas as pd
from pathlib import Path

# ===== ПАРАМЕТРЫ =====
TICKER = os.getenv("TICKER", "SBER")
TF     = os.getenv("TF", "5m")
H      = int(os.getenv("H", "36"))                 # горизонт в барах
TAU    = float(os.getenv("TAU", "0.02"))           # порог для матемодели (доля)

START  = os.getenv("START", "2025-10-11")          # включительно (UTC)
END    = os.getenv("END",   "2025-10-21")          # включительно (UTC)

# шаг по барам и прогрев индикаторов
STEP_BARS = int(os.getenv("STEP_BARS", "1"))       # 1 = каждый бар; 5,12 = разрежение
WARMUP    = int(os.getenv("WARMUP", "64"))         # минимум баров перед первым сигналом

# пути проекта (подстрой под свою структуру)
FEAT_LABELED = os.getenv("FEAT_LABELED", f"../../data/features/ml/{TICKER}/{TF}/features_labeled.parquet")
FEAT_TODAY   = os.getenv("FEAT_TODAY",   f"../../data/features/ml/{TICKER}/{TF}/features_today.parquet")
SIGNALS_CSV  = os.getenv("SIGNALS_CSV",  f"../../data/signals/{TICKER}_{TF}.csv")
ML_PKL       = os.getenv("ML_PKL",       f"../../models/{TICKER}_{TF}_lgbm.pkl")

# CV
CV_MODEL     = os.getenv("CV_MODEL",     f"../../models/cv_resnet18_state.pt")  # state_dict .pt
CV_TMP_DIR   = os.getenv("CV_TMP_DIR",   f"../../data/cv/tmp/{TICKER}/{TF}")
CV_SIZE      = int(os.getenv("CV_SIZE", "64"))
CV_STEP      = int(os.getenv("CV_STEP", "16"))
CV_IMG_ROOT  = os.getenv("CV_IMG_ROOT", "")  # опц. корень для PNG (если манифест хранит относительные пути)

# === ваш inference ===
# infer_ml(ml_pkl, feat_today_path, H) -> (dict,status)
# infer_math_from_prices(feat_today_path, H_min, thr_pct) -> (dict,status)
from inference import infer_ml, infer_math_from_prices

# рендер PNG + manifest
from scripts.render_cv_images import render_with_manifest

def ensure_parent(p: str | Path):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1])*60
    if tf.endswith("d"): return int(tf[:-1])*60*24
    raise ValueError(f"unsupported TF: {tf}")

TF_MIN = tf_to_minutes(TF)

def load_labeled() -> pd.DataFrame:
    df = pd.read_parquet(FEAT_LABELED).sort_values("time").reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number, "bool"]).columns
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(np.uint8)
    df[num_cols] = (df[num_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .ffill().fillna(0.0))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df

def slice_day(df: pd.DataFrame, day_utc: pd.Timestamp) -> pd.DataFrame:
    m = (df["time"] >= day_utc) & (df["time"] < day_utc + pd.Timedelta(days=1))
    return df.loc[m].copy()

def append_row(row: dict):
    ensure_parent(SIGNALS_CSV)
    base = pd.read_csv(SIGNALS_CSV) if Path(SIGNALS_CSV).exists() else pd.DataFrame()
    base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
    base["time"] = pd.to_datetime(base["time"], utc=True)
    base = base.drop_duplicates(subset=["time"], keep="last").sort_values("time")
    base.to_csv(SIGNALS_CSV, index=False)

def daterange_inclusive(start: str, end: str):
    d0 = pd.Timestamp(start, tz="UTC").normalize()
    d1 = pd.Timestamp(end,   tz="UTC").normalize()
    cur = d0
    while cur <= d1:
        yield cur
        cur += pd.Timedelta(days=1)

# --- утилита резолвинга путей PNG из манифеста ---
def _resolve_path(p: str, man_csv: str) -> Path:
    q = Path(p)
    if q.is_file(): return q.resolve()
    # кандидаты:
    cands = []
    if CV_IMG_ROOT:
        cands.append(Path(CV_IMG_ROOT) / p)
    cands += [
        Path.cwd() / p,
        Path(__file__).resolve().parent / p,
        (Path(__file__).resolve().parent / ".." / ".." / p),
        Path(man_csv).parent / p,
    ]
    for c in cands:
        c = c.resolve()
        if c.is_file(): return c
    return q  # вернуть как есть; дальше отфильтруем

# ===== CV блок: рендер дня и инференс по кадрам =====
def compute_cv_scores_for_day(dfd: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает DataFrame ['time','s_cv'].
    time = t_end кадра, s_cv = P(up)-P(down).
    """
    if not Path(CV_MODEL).exists():
        return pd.DataFrame(columns=["time","s_cv"])

    ensure_parent(CV_TMP_DIR + "/manifest.csv")
    man_csv = str(Path(CV_TMP_DIR) / "manifest.csv")

    # FEAT_TODAY уже сохранён целиком (ниже в main), рендерим день
    try:
        render_with_manifest(
            in_parquet = FEAT_TODAY,
            ticker     = TICKER,
            tf         = TF,
            size       = CV_SIZE,
            step       = CV_STEP,
            horizon    = H,
            show_axes  = False,
            show_volume= True,
            vol_mode   = "zscore",
            fixed_ylim = "window",
            ylim_pad   = 0.02,
            ohlc_mode  = "close+hl_wick",
            extra_ch   = "atr_rel" if "atr_rel" in dfd.columns else None,
            manifest_csv = man_csv,
            seed = 42
        )
    except Exception as e:
        print(f"[warn] CV render failed: {e}")
        return pd.DataFrame(columns=["time","s_cv"])

    mdf = pd.read_csv(man_csv)
    if mdf.empty or "t_end" not in mdf or "path" not in mdf:
        return pd.DataFrame(columns=["time","s_cv"])

    # фикс путей
    mdf["abs_path"] = mdf["path"].astype(str).map(lambda s: _resolve_path(s, man_csv))
    mdf = mdf[mdf["abs_path"].map(lambda p: Path(p).is_file())].copy()
    if mdf.empty:
        return pd.DataFrame(columns=["time","s_cv"])

    # модель и препроцесс
    import torch, torchvision as tv
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tv.models.resnet18(weights=None, num_classes=3)
    model.conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    sd = torch.load(CV_MODEL, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    T = tv.transforms.Compose([
        tv.transforms.Grayscale(1),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.5],[0.5]),
    ])

    # батч-инференс
    probs = []
    B = 512
    paths = mdf["abs_path"].astype(str).tolist()
    with torch.no_grad():
        buf = []
        for pth in paths:
            try:
                img = Image.open(pth).convert("L")
                buf.append(T(img))
            except Exception:
                buf.append(None)
            if len(buf) == B:
                xb = torch.stack([x for x in buf if x is not None], 0).to(device)
                if xb.shape[0] > 0:
                    pb = torch.softmax(model(xb), dim=1).cpu().numpy()
                    probs.append(pb)
                buf = []
        if buf:
            xb = torch.stack([x for x in buf if x is not None], 0).to(device)
            if xb.shape[0] > 0:
                pb = torch.softmax(model(xb), dim=1).cpu().numpy()
                probs.append(pb)

    if not probs:
        return pd.DataFrame(columns=["time","s_cv"])
    P = np.vstack(probs)
    s_cv = (P[:,2] - P[:,0]).astype(float)

    out = pd.DataFrame({
        "time": pd.to_datetime(mdf["t_end"], utc=True, errors="coerce"),
        "s_cv": s_cv
    }).dropna().sort_values("time").reset_index(drop=True)
    return out

# ===== MAIN =====
if __name__ == "__main__":
    if not Path(FEAT_LABELED).exists(): raise FileNotFoundError(FEAT_LABELED)
    if not Path(ML_PKL).exists(): raise FileNotFoundError(ML_PKL)

    df_all = load_labeled()
    total_added = 0

    for day_utc in daterange_inclusive(START, END):
        dfd = slice_day(df_all, day_utc)
        if dfd.empty:
            print(f"[skip] {day_utc.date()}: нет данных")
            continue

        # сохраним «весь день» в FEAT_TODAY
        ensure_parent(FEAT_TODAY)
        dfd.to_parquet(FEAT_TODAY, index=False)

        # CV по дню
        cv_df = compute_cv_scores_for_day(dfd)
        cv_times = cv_df["time"].to_numpy() if not cv_df.empty else np.array([], dtype="datetime64[ns]")
        cv_vals  = cv_df["s_cv"].to_numpy()  if not cv_df.empty else np.array([], dtype=float)

        # по барам
        added = 0
        for i in range(max(WARMUP-1, 0), len(dfd), STEP_BARS):
            cur = dfd.iloc[:i+1].copy()
            cur.to_parquet(FEAT_TODAY, index=False)

            # инференс ML + Math
            ml_out, _   = infer_ml(ML_PKL, FEAT_TODAY, H)
            math_out, _ = infer_math_from_prices(FEAT_TODAY, H_min=H*TF_MIN, thr_pct=TAU)
            ml_out   = ml_out   or {}
            math_out = math_out or {}

            t_i = pd.to_datetime(cur["time"].iloc[-1], utc=True)

            # ближайший прошедший кадр CV
            s_cv_val = np.nan
            if cv_times.size:
                j = cv_times.searchsorted(t_i.to_datetime64(), side="right") - 1
                if j >= 0:
                    s_cv_val = float(cv_vals[j])

            row = dict(
                time = t_i,
                close= float(cur["close"].iloc[-1]),
                s_ml = float(ml_out.get("score", np.nan)),
                s_cv = s_cv_val,
                p_up = float(ml_out.get("P_up", np.nan)),
                p_down=float(ml_out.get("P_down", np.nan)),
                p_up_tau=float(math_out.get("p_up_tau", np.nan)),
            )
            append_row(row)
            added += 1
            total_added += 1

        print(f"[ok] {day_utc.date()}: bars={len(dfd)} -> signals_added={added}, cv_frames={0 if cv_df is None else len(cv_df)}")

    print(f"[done] signals -> {SIGNALS_CSV}, added={total_added}")
