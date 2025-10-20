# infer_today.py
# pip install joblib lightgbm xgboost torch torchvision pandas pyarrow pillow numpy
import os, math, joblib, numpy as np, pandas as pd
from pathlib import Path

# ====== НАСТРОЙКИ ======
TICKER   = "SBER"
TF       = "5m"
H        = 36                 # для 5m и горизонта 180 минут
THRESH_PCT = 0.02            # порог для матмодели (±2%)

# Пути данных за сегодня (сформируйте заранее вашим пайплайном)
FEAT_TODAY = f"../data/features/ml/{TICKER}/{TF}/features_today.parquet"   # табличные фичи за сегодня
CV_MANIFEST= f"../data/cv/images/{TICKER}/{TF}/today_manifest.csv"         # манифест PNG за сегодня (опц.)

# Пути моделей (уже обучены и сохранены)
ML_MODEL_PKL = "../models/SBER_5m_lgbm.pkl"         # joblib.dump(...) или pickle.dump(...)
CV_MODEL_PKL = "../models/cv_resnet18.pt"       # torch.save(model, ...)
# ==========================================


# ---------- ВСПОМОГАТЕЛЬНЫЕ ----------
def _clean_X(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if c.startswith(("y_tb_", "ret_fwd_"))] + ["time"]
    X = df.drop(columns=drop_cols, errors="ignore")
    # выкинуть object-колонки; bool -> uint8
    obj = X.select_dtypes(include=["object"]).columns
    if len(obj): X = X.drop(columns=obj)
    for c in X.select_dtypes(include=["bool"]).columns:
        X[c] = X[c].astype(np.uint8)
    # заменить inf/NaN, ffill без утечек
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return X

def _align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        # добьем недостающие нулями и упорядочим
        for col in names:
            if col not in X.columns: X[col] = 0.0
        X = X[names]
    return X

def _softmax(z):
    z = np.asarray(z, dtype=float)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ---------- 1) ML-инференс (табличная) ----------
def infer_ml(model_path: str, feat_path: str, H: int):
    if not Path(feat_path).is_file():
        return None, "no_features_today"

    df = pd.read_parquet(feat_path).sort_values("time").reset_index(drop=True)
    X = _clean_X(df)
    model = joblib.load(model_path)  # LightGBM/CatBoost/XGB, сохранённые в .pkl
    X = _align_features(model, X)

    # ожидаем proba 3-класса: [down, flat, up]; fallback для decision_function
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)  # shape [N,3]
    else:
        # если вы сохранили пайплайн с Calibrated + LR над базовой моделью: передайте нужный вход
        logits = model.predict(X)
        proba = _softmax(logits)

    p_dn, p_fl, p_up = proba[:,0], proba[:,1], proba[:,2]
    score = p_up - p_dn
    # берём последний бар
    last = dict(P_down=float(p_dn[-1]), P_flat=float(p_fl[-1]), P_up=float(p_up[-1]), score=float(score[-1]))
    return last, "ok"

# ---------- 2) CV-инференс (ResNet, torch.save(model, ...)) ----------
def infer_cv(model_path: str, manifest_csv: str):
    import os, numpy as np, pandas as pd
    from pathlib import Path
    import torch, torchvision as tv
    from PIL import Image, UnidentifiedImageError

    if not Path(model_path).is_file() or not Path(manifest_csv).is_file():
        return None, "no_cv_inputs"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- загрузка модели/весов ---
    def load_torch_model(pt_path: str, device):
        meta = {}
        # 1) безопасно: state_dict (weights_only=True)
        try:
            obj = torch.load(pt_path, map_location=device, weights_only=True)
            if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
                sd = obj
            elif isinstance(obj, dict) and "state_dict" in obj:
                sd, meta = obj["state_dict"], obj.get("meta", {})
            else:
                raise RuntimeError("not a pure state_dict")
            in_ch  = int(meta.get("in_channels", 1))
            n_cls  = int(meta.get("num_classes", 3))
            arch   = meta.get("arch", "resnet18")
            if arch != "resnet18":
                raise NotImplementedError(f"unsupported arch: {arch}")
            m = tv.models.resnet18(weights=None, num_classes=n_cls)
            if in_ch != 3:
                m.conv1 = torch.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if unexpected:
                # допустимы расхождения типа 'fc.*' при другом числе классов
                pass
            return m.to(device).eval(), meta
        except Exception:
            # 2) legacy: целая модель/бандл (доверенный файл)
            obj = torch.load(pt_path, map_location=device, weights_only=False)
            if hasattr(obj, "state_dict") and not isinstance(obj, dict):
                m = obj.to(device).eval()
                meta = getattr(obj, "meta", {})
                return m, meta
            if isinstance(obj, dict):
                sd = obj.get("state_dict", obj)
                meta = obj.get("meta", {})
                in_ch  = int(meta.get("in_channels", 1))
                n_cls  = int(meta.get("num_classes", 3))
                arch   = meta.get("arch", "resnet18")
                if arch != "resnet18":
                    raise NotImplementedError(f"unsupported arch: {arch}")
                m = tv.models.resnet18(weights=None, num_classes=n_cls)
                if in_ch != 3:
                    m.conv1 = torch.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
                m.load_state_dict(sd, strict=False)
                return m.to(device).eval(), meta
            raise RuntimeError("unsupported checkpoint format")

    try:
        model, meta = load_torch_model(model_path, device)
    except Exception as e:
        return None, f"load_failed: {e}"

    in_ch = int(meta.get("in_channels", 1))
    # --- манифест ---
    dfm = pd.read_csv(manifest_csv)
    if "path" not in dfm:
        return None, "bad_manifest"
    dfm = dfm[dfm["path"].map(lambda p: Path(p).is_file())].copy()
    if dfm.empty:
        return None, "empty_manifest"

    if "t_end" in dfm.columns:
        dfm["t_end"] = pd.to_datetime(dfm["t_end"], errors="coerce")
        dfm = dfm.dropna(subset=["t_end"]).sort_values("t_end")

    sample = dfm.tail(min(64, len(dfm)))  # последние до 64 файлов

    # --- трансформации ---
    if in_ch == 1:
        T = tv.transforms.Compose([
            tv.transforms.Grayscale(1),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5],[0.5]),
        ])
    else:  # 3 канала
        T = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    imgs = []
    kept = []
    for p in sample["path"]:
        try:
            img = Image.open(p).convert("L" if in_ch==1 else "RGB")
            imgs.append(T(img))
            kept.append(p)
        except (UnidentifiedImageError, OSError):
            continue

    if not imgs:
        return None, "no_images_loaded"

    X = torch.stack(imgs, 0).to(device)  # [B,C,64,64]

    with torch.no_grad():
        logits = model(X)
        proba  = torch.softmax(logits, dim=1).detach().cpu().numpy()

    if proba.shape[1] != 3:
        return None, f"unexpected_num_classes:{proba.shape[1]}"

    p_dn, p_fl, p_up = proba[:,0], proba[:,1], proba[:,2]
    score = p_up - p_dn

    agg = dict(
        P_down=float(p_dn.mean()),
        P_flat=float(p_fl.mean()),
        P_up=float(p_up.mean()),
        score=float(score.mean()),
        last_P_up=float(p_up[-1]),
        last_score=float(score[-1]),
        n=len(kept)
    )
    return agg, "ok"

# ---------- 3) Матмодель (GBM + EWMA) ----------
def ewma_sigma(r, lam=0.94):
    s2 = np.zeros_like(r, dtype=float)
    s2[0] = np.var(r[:100]) if len(r)>100 else np.var(r)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t-1] + (1-lam) * (r[t-1]**2)
    return np.sqrt(s2)

def gbm_mc(mu_yr, sigma_yr, S0, minutes, paths=20000):
    # перевод минут в долю года: 252 торговых дня * ~6.5ч * 60м
    dt_year = minutes / (252*6.5*60)
    Z = np.random.normal(size=paths)
    ST = S0 * np.exp((mu_yr - 0.5*sigma_yr**2)*dt_year + sigma_yr*np.sqrt(dt_year)*Z)
    return ST/S0 - 1.0

def infer_math_from_prices(feat_today_path: str, H_min: int, thr_pct: float):
    if not Path(feat_today_path).is_file():
        return None, "no_prices"

    df = pd.read_parquet(feat_today_path).sort_values("time").reset_index(drop=True)
    if "close" not in df.columns or len(df) < 10:
        return None, "no_close"

    close = df["close"].astype(float).to_numpy()
    logret = np.diff(np.log(close))
    if len(logret) < 50:
        return None, "too_short"

    sigma_t = ewma_sigma(logret)
    # локальный минутный дрейф → годовой
    mu_min = pd.Series(logret).rolling(78, min_periods=20).mean().to_numpy()  # 78 пятиминуток ~тр. день
    mu_yr  = float(mu_min[-1] * (252*6.5*60))
    sigma_yr = float(sigma_t[-1] * math.sqrt(252*6.5*60))

    ret = gbm_mc(mu_yr, sigma_yr, close[-1], minutes=H_min, paths=50000)
    VaR95  = float(np.quantile(ret, 0.05))
    CVaR95 = float(ret[ret <= VaR95].mean())
    p_up_tau = float((ret > thr_pct).mean())
    score = float(mu_yr / (sigma_yr + 1e-12))
    return dict(p_up_tau=p_up_tau, VaR95=VaR95, CVaR95=CVaR95,
                mu_yr=mu_yr, sigma_yr=sigma_yr, score=score), "ok"


# ========================= MAIN =========================
if __name__ == "__main__":
    # 1) ML
    ml_out, ml_status = infer_ml(ML_MODEL_PKL, FEAT_TODAY, H)
    print("[ML]", ml_status, ml_out)

    # 2) CV
    try:
        cv_out, cv_status = infer_cv(CV_MODEL_PKL, CV_MANIFEST)
    except Exception as e:
        cv_out, cv_status = None, f"error: {e}"
    print("[CV]", cv_status, cv_out)

    # 3) Math
    math_out, math_status = infer_math_from_prices(FEAT_TODAY, H*5, THRESH_PCT)  # H*5 минут т.к. TF=5m
    print("[MATH]", math_status, math_out)

    # 4) Итоговый сигнал (если есть два источника вероятностей)
    def pick(x, key, default=np.nan):
        return (x or {}).get(key, default)
    s_ml  = pick(ml_out, "score")
    s_cv  = pick(cv_out, "score")
    # простейший ансамбль: среднее
    if not np.isnan(s_ml) or not np.isnan(s_cv):
        ss = np.nanmean([s_ml, s_cv])
        print(f"[SIGNAL] s_ensemble={ss:.4f}  "
              f"(s_ml={s_ml if not np.isnan(s_ml) else 'NaN'}, s_cv={s_cv if not np.isnan(s_cv) else 'NaN'})")
