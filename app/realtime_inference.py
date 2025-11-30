"""
Realtime inference runner (только live-режим).

Каждые 5 минут:
- подтягивает новые данные с Tinkoff Invest API,
- прогоняет ML (SSM) + CV (ResNet50 по свечному графику) + RL (PPO),
- включает продвинутый риск-менеджмент (HMM, GARCH, ARIMA, Monte-Carlo, Kelly),
- принимает решение по последней свече (open/no_trade_*),
- печатает всё в консоль.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import now

# --- optional dependencies (ML/CV/визуализация) ---
try:  # ML
    import xgboost as xgb  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore

try:  # CV (torch + torchvision)
    import torch
    from torch import nn
    from torchvision import transforms, models as tv_models
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    transforms = None  # type: ignore
    tv_models = None  # type: ignore

try:  # Candle rendering
    import mplfinance as mpf
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    mpf = None  # type: ignore
    plt = None  # type: ignore

# --- RL части ---
from trained_models.RL import build_feature_frame
from trained_models.RL.agent import RLAgent
from trained_models.RL.env import PortfolioState

# --- Новая SSM-модель ---
from trained_models.ssm_model_for_new_data import predict_new_data as ssm_predict_new_data

# --- базовые настройки и токен ---

DEFAULT_TOKEN = "write your ticker here, .env not works beacuse of encoding issues"
DEFAULT_TZ = pytz.timezone("Europe/Moscow")
DEFAULT_TIMEFRAME = CandleInterval.CANDLE_INTERVAL_5_MIN
INTERVAL_SEC = 5 * 60  # 5 минут
# Флаг отладочного вывода
DEBUG = False

SSM_SEQ_LEN = 60  # как в конфиге модели

TICKER = "SBER"
LOOKBACK = 60
WINDOW = 32

INITIAL_BALANCE = 100_000.0
DAYS_FOR_FETCH = 10

ML_MODEL_PATH="app/trained_models/best_ssm_model.pth"
CV_MODEL_PATH="app/trained_models/best_cv_model.pth"
META_CV_PATH="app/trained_models/label_map.json"
RL_MODEL_PATH="app/trained_models/rl/checkpoints/ppo_trading_agent.zip"

# Пороги для лайв-решения
ML_PROBA_THR_LIVE = 0.8
CV_PROBA_THR_LIVE = 0.8

# Константы продвинутого риск-менеджмента
BASE_THR_OVERLAY = 0.60      # базовый порог для агрегированной вероятности направления
VOL_PCTL_ABS = 0.95          # верхний квантиль волатильности, выше которого не торгуем
MC_H = 20                    # горизонт в шагах для Monte-Carlo
MC_MARGIN_HIT = 0.05         # p(TP) должно быть хотя бы на 5% выше p(SL)
TARGET_VOL = 0.02            # таргет по волатильности портфеля (2% условно)
F_MAX = 0.10                 # максимальная доля капитала в позиции
TP_ATR_K = 5.0               # TP = close ± TP_ATR_K * ATR
SL_ATR_K = 2.0               # SL = close ∓ SL_ATR_K * ATR


# =====================================================================
#                 Утилиты для загрузки данных Tinkoff
# =====================================================================

def fetch_tinkoff_candles(
    token: str,
    ticker: str,
    days: int = 1,
    interval: CandleInterval = DEFAULT_TIMEFRAME,
    tz: pytz.BaseTzInfo = DEFAULT_TZ,
) -> pd.DataFrame:
    """Download OHLCV candles from the Tinkoff Invest API."""
    if not token:
        raise RuntimeError("TINKOFF_TOKEN is required to fetch candles")

    with Client(token) as client:
        instruments: InstrumentsService = client.instruments
        shares = instruments.shares().instruments
        figi: Optional[str] = None
        for share in shares:
            if share.ticker.upper() == ticker.upper():
                figi = share.figi
                break
        if not figi:
            raise RuntimeError(f"FIGI for {ticker} not found")

        end = now()
        start = end - timedelta(days=days)
        candles = client.get_all_candles(figi=figi, from_=start, to=end, interval=interval)

        data = [
            {
                "time": candle.time.astimezone(tz),
                "open": candle.open.units + candle.open.nano / 1e9,
                "high": candle.high.units + candle.high.nano / 1e9,
                "low": candle.low.units + candle.low.nano / 1e9,
                "close": candle.close.units + candle.close.nano / 1e9,
                "volume": candle.volume,
            }
            for candle in candles
        ]

    return pd.DataFrame(data)


def compute_atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """True Range + Wilder's ATR."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr


# =====================================================================
#            Продвинутые функции риск-менеджмента (overlays)
# =====================================================================

def fit_hmm_regimes(returns: pd.Series, n_states: int = 2):
    """
    HMM по доходностям для выявления режимов (спокойный/турбулентный).

    returns — Series лог-доходностей.
    Возвращает (hmm_model | None, regimes Series | None).
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except ImportError:
        return None, None

    X = returns.dropna().values.reshape(-1, 1)
    if len(X) < 200:
        return None, None

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    hmm.fit(X)
    hidden = pd.Series(hmm.predict(X), index=returns.dropna().index)
    return hmm, hidden.reindex(returns.index)


def garch_sigma(returns: pd.Series) -> pd.Series:
    """
    Оценка условной волатильности через GARCH(1,1).

    returns — лог-доходности.
    Возвращает Series sigma_t.
    """
    try:
        from arch import arch_model  # type: ignore
    except ImportError:
        # fallback: скользящий std
        return returns.abs().rolling(20).std().reindex(returns.index)

    r = (returns.dropna() * 100).astype(float)  # проценты
    if len(r) < 300:
        return returns.abs().rolling(20).std().reindex(returns.index)

    am = arch_model(r, p=1, q=1, mean="Constant", vol="GARCH", dist="normal")
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100.0  # обратно в доли
    return cond_vol.reindex(returns.index)


def arima_sign(returns: pd.Series) -> pd.Series:
    """
    Грубый ARIMA-сигнал: знак прогноза следующей доходности.

    Возвращает Series с константным значением +1/-1 (последний прогноз).
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    except ImportError:
        return pd.Series(index=returns.index, dtype=float)

    y = returns.dropna()
    if len(y) < 100:
        return pd.Series(index=returns.index, dtype=float)

    model = ARIMA(y, order=(1, 0, 1))
    res = model.fit()
    fc = res.forecast(1)
    sgn = np.sign(fc.iloc[0])
    return pd.Series(sgn, index=returns.index).ffill()


def mc_hit_probs(
    close_series: pd.Series,
    mu: float,
    sigma: float,
    H: int,
    n_paths: int = 2000,
    dt: float = 1.0,
):
    """
    Monte-Carlo по геометрическому броуновскому движению для цены.

    Возвращает (max_arr, min_arr) по всем путям, чтобы можно было оценить
    вероятности достижения TP и SL.
    """
    S0 = float(close_series.iloc[-1])
    z = np.random.normal(size=(n_paths, H))
    steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    paths = S0 * np.exp(np.cumsum(steps, axis=1))
    max_arr = paths.max(axis=1)
    min_arr = paths.min(axis=1)
    return max_arr, min_arr


def kelly_lite(prob_up: float, R: float = 1.0, f_max: float = 0.1) -> float:
    """
    Kelly-доля для бинарного исхода (up/down), ограниченная f_max.
    """
    p = np.clip(prob_up, 1e-6, 1 - 1e-6)
    f = (p * R - (1 - p)) / R
    return float(np.clip(f, 0.0, f_max))


def apply_risk_overlays(
    hist: pd.DataFrame,
    direction: int,
    close_price: float,
    tp_price: float,
    sl_price: float,
    proba_up: float,
    proba_down: float,
    rl_size: float,
    base_thr_overlay: float = BASE_THR_OVERLAY,
    vol_pctl_abs: float = VOL_PCTL_ABS,
    mc_h: int = MC_H,
    mc_margin_hit: float = MC_MARGIN_HIT,
    target_vol: float = TARGET_VOL,
    f_max: float = F_MAX,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Применяет все продвинутые оверлеи риск-менеджмента к уже согласованному направлению
    (ML+CV+RL) и рассчитанным TP/SL.

    На вход:
        hist        — DataFrame со столбцами 'close', 'atr_14' (и, при необходимости, др.)
        direction   — +1 (long) или -1 (short)
        close_price — текущая цена
        tp_price    — рассчитанный TP
        sl_price    — рассчитанный SL
        proba_up    — агрегированная вероятность направления up
        proba_down  — агрегированная вероятность направления down
        rl_size     — размер позиции, предложенный RL (0..1)

    Возвращает:
        (decision, size_frac):
            decision  — dict c no_trade_* если нужно НЕ торговать (иначе None),
            size_frac — финальная доля капитала под сделку (0..f_max), если trade допустим.
    """
    # Гарантируем наличие лог-доходностей
    hist = hist.copy()
    if "logret" not in hist.columns:
        hist["logret"] = np.log(hist["close"]).diff()

    # HMM режим
    try:
        _, regimes = fit_hmm_regimes(hist["logret"])
        regime_last = regimes.iloc[-1] if regimes is not None else np.nan
    except Exception:
        regime_last = np.nan

    # GARCH-вола
    try:
        sigma_series = garch_sigma(hist["logret"])
    except Exception:
        sigma_series = hist["logret"].rolling(20).std()

    sigma_last = float(sigma_series.iloc[-1])
    vol_cut = float(sigma_series.quantile(vol_pctl_abs))

    # Очень высокая волатильность → не торгуем
    if np.isfinite(sigma_last) and sigma_last >= vol_cut and sigma_last > 0:
        decision = {
            "type": "no_trade_high_vol",
            "reason": f"sigma_last={sigma_last:.4f} >= vol_cut={vol_cut:.4f}",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return decision, 0.0

    # ARIMA sign
    try:
        arima_series = arima_sign(hist["logret"])
        arima_last = float(arima_series.iloc[-1])
    except Exception:
        arima_last = np.nan

    # Порог по агрегированной вероятности в зависимости от режима
    thr = base_thr_overlay
    try:
        if not np.isnan(regime_last) and int(regime_last) == 1:
            thr += 0.05  # в "плохом" режиме требуем выше уверенность
    except Exception:
        pass

    # Выбираем вероятность направления по знаку
    proba_dir = proba_up if direction > 0 else proba_down

    if np.isfinite(proba_dir) and proba_dir < thr:
        decision = {
            "type": "no_trade_overlay_low_prob",
            "reason": f"aggregated prob_dir={proba_dir:.4f} < thr={thr:.4f}, regime={regime_last}",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return decision, 0.0

    # ARIMA фильтр: если ARIMA против направления и преимущество вероятности маленькое — не торгуем
    if np.isfinite(arima_last):
        if direction > 0 and arima_last < 0 and proba_up < proba_down + 0.05:
            decision = {
                "type": "no_trade_arima_filter",
                "reason": f"ARIMA sign={arima_last} против long, proba_up={proba_up:.4f}, proba_down={proba_down:.4f}",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            }
            return decision, 0.0
        if direction < 0 and arima_last > 0 and proba_down < proba_up + 0.05:
            decision = {
                "type": "no_trade_arima_filter",
                "reason": f"ARIMA sign={arima_last} против short, proba_up={proba_up:.4f}, proba_down={proba_down:.4f}",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            }
            return decision, 0.0

    # Monte-Carlo: вероятность достижения TP vs SL
    try:
        if not np.isfinite(sigma_last) or sigma_last <= 0:
            sigma_last_eff = float(hist["logret"].rolling(20).std().iloc[-1] or 1e-6)
        else:
            sigma_last_eff = sigma_last

        mu_loc = float(hist["logret"].iloc[-50:].mean())
        max_arr, min_arr = mc_hit_probs(
            hist["close"], mu=mu_loc, sigma=sigma_last_eff, H=mc_h, n_paths=1500
        )

        if direction > 0:
            p_hit_tp = float((max_arr >= tp_price).mean())
            p_hit_sl = float((min_arr <= sl_price).mean())
        else:
            p_hit_tp = float((min_arr <= tp_price).mean())
            p_hit_sl = float((max_arr >= sl_price).mean())

        if p_hit_tp < p_hit_sl + mc_margin_hit:
            decision = {
                "type": "no_trade_mc_filter",
                "reason": f"p_hit_tp={p_hit_tp:.3f} < p_hit_sl+margin={p_hit_sl + mc_margin_hit:.3f}",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            }
            return decision, 0.0
    except Exception as e:
        if DEBUG:
            print("Monte-Carlo overlay error:", e)

    # Kelly + ограничение по волатильности + RL
    # отношение профит/риск в ценах
    R = abs((tp_price - close_price) / max(abs(close_price - sl_price), 1e-9))

    prob_dir = proba_up if direction > 0 else proba_down
    if not np.isfinite(prob_dir):
        prob_dir = 0.5  # fallback

    f_kelly = kelly_lite(prob_dir, R=R, f_max=f_max)

    # size_vol: таргетируем волатильность
    sigma_last_eff = sigma_last if np.isfinite(sigma_last) and sigma_last > 0 else float(
        hist["logret"].rolling(20).std().iloc[-1] or 1e-6
    )
    size_vol = target_vol / max(sigma_last_eff, 1e-6)

    size_kelly = float(np.clip(min(f_kelly, size_vol), 0.0, f_max))
    size_from_rl = 0.15 * float(rl_size)

    size_frac = float(np.clip(min(size_kelly, size_from_rl), 0.0, f_max))

    if size_frac <= 0.0:
        decision = {
            "type": "no_trade_zero_size",
            "reason": f"size_frac computed as 0 (kelly={size_kelly:.4f}, rl={size_from_rl:.4f})",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return decision, 0.0

    # успех: торговля разрешена, size_frac > 0
    return None, size_frac


# =====================================================================
#                           SSM (ML) часть
# =====================================================================

def compute_ssm_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Готовим входные фичи для SSM-модели.
    Ожидается: time, open, high, low, close, volume.
    """
    df = df.sort_values("time").reset_index(drop=True).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    df["sma_ratio"] = df["close"] / df["sma_10"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window_rsi = 14
    avg_gain = gain.rolling(window_rsi).mean()
    avg_loss = loss.rolling(window_rsi).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    window_bb = 20
    df["boll_mid"] = df["close"].rolling(window_bb).mean()
    df["boll_std"] = df["close"].rolling(window_bb).std()
    df["boll_up"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_low"] = df["boll_mid"] - 2 * df["boll_std"]
    df["boll_pos"] = (df["close"] - df["boll_mid"]) / df["boll_std"].replace(0, np.nan)

    df["momentum_5"] = df["close"] / df["close"].shift(5) - 1.0
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1.0

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["atr"] = compute_atr_wilder(df, n=14)

    return df


def predict_ml_ssm(
    df_candles: pd.DataFrame,
    model_path: str = "trained_models/best_ssm_model.pth",
    return_proba: bool = True,
) -> Dict[str, Any]:
    """
    Запуск SSM-модели на свечном DataFrame df_candles.

    Возвращает:
        {
          "indices": [int...],   # 0=down, 1=flat, 2=up
          "labels":  [str...],   # "down"/"flat"/"up"
          "proba":   np.ndarray  # (N, 3) [P(down), P(flat), P(up)]
        }
    """
    df_feat = compute_ssm_base_features(df_candles.copy())

    if "label" not in df_feat.columns:
        df_feat["label"] = "flat"

    tmp_path = "_tmp_ssm_input.csv"
    df_feat.to_csv(tmp_path, index=False)

    res_df = ssm_predict_new_data(model_path, tmp_path, sequence_length=SSM_SEQ_LEN)

    if res_df is None or len(res_df) == 0:
        result: Dict[str, Any] = {
            "indices": [],
            "labels": [],
        }
        if return_proba:
            result["proba"] = np.zeros((0, 3), dtype=float)
        return result

    probs = res_df[["prob_down", "prob_flat", "prob_up"]].to_numpy()

    label_to_idx = {"down": 0, "flat": 1, "up": 2}
    labels = res_df["predicted_label"].tolist()
    indices = [label_to_idx.get(lbl, 1) for lbl in labels]

    result: Dict[str, Any] = {"indices": indices, "labels": labels}
    if return_proba:
        result["proba"] = probs

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return result


# =====================================================================
#                      CV: candlestick + ResNet50
# =====================================================================

def add_cv_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA и Bollinger bands для более информативного графика."""
    df = df.copy()
    df["ema_10"] = df["close"].ewm(span=10).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    mid = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["boll_up"] = mid + 2 * std
    df["boll_low"] = mid - 2 * std
    return df


@lru_cache(maxsize=1)
def get_mpf_style():
    """Кэшируем стиль для mplfinance, чтобы не пересоздавать каждый раз."""
    if mpf is None or plt is None:
        raise ImportError(
            "mplfinance and matplotlib must be installed; "
            "install via `pip install mplfinance matplotlib`"
        )

    mc = mpf.make_marketcolors(
        up="lime", down="red", edge="white", wick="white", volume="gray"
    )
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        facecolor="black",
        edgecolor="white",
        marketcolors=mc,
        rc={"axes.labelcolor": "white", "axes.edgecolor": "white"},
    )
    return style


def render_candle_image(
    sub_df: pd.DataFrame,
    img_size: tuple = (8, 4),
    dpi: int = 100,
    use_jpg: bool = True,  # оставлен для совместимости
    style: Optional[Any] = None,
) -> np.ndarray:
    """Рендерит окно свечей в RGB-картинку (NumPy)."""
    if mpf is None or plt is None:
        raise ImportError(
            "mplfinance and matplotlib must be installed to render candle images; "
            "install via `pip install mplfinance matplotlib`"
        )

    if style is None:
        style = get_mpf_style()

    fig, axes = mpf.plot(
        sub_df,
        type="candle",
        style=style,
        volume=True,
        figsize=img_size,
        tight_layout=True,
        show_nontrading=True,
        returnfig=True,
    )

    for ax in axes:
        ax.set_axis_off()
        ax.grid(False)
    if "ema_10" in sub_df.columns:
        axes[0].plot(sub_df.index, sub_df["ema_10"], linewidth=1)
    if "ema_20" in sub_df.columns:
        axes[0].plot(sub_df.index, sub_df["ema_20"], linewidth=1)
    if {"boll_up", "boll_low"}.issubset(sub_df.columns):
        axes[0].plot(sub_df.index, sub_df["boll_up"], linestyle="--", linewidth=0.8)
        axes[0].plot(sub_df.index, sub_df["boll_low"], linestyle="--", linewidth=0.8)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))[..., :3]
    plt.close(fig)
    return img


@lru_cache(maxsize=1)
def get_cv_transform():
    """Кэшируем трансформы для CV-модели."""
    if torch is None or transforms is None:
        raise ImportError("torch and torchvision must be installed to prepare image tensors")

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ]
    )


def prepare_cv_tensor(img: np.ndarray):
    """RGB → тензор для CV-модели."""
    if torch is None or transforms is None:
        raise ImportError("torch and torchvision must be installed to prepare image tensors")

    transform = get_cv_transform()
    return transform(img).unsqueeze(0)


def create_cv_model_resnet50(num_classes: int, device: Optional[str] = None):
    """
    ResNet50 + кастомная голова, как в cv_model.ipynb.
    Используется только для инференса.
    """
    if torch is None or nn is None or tv_models is None:
        raise ImportError("torch/torchvision must be installed to create the CV model")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
    in_feat = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Linear(512, num_classes),
    )

    model.to(device)
    model.eval()
    return model


@lru_cache(maxsize=8)
def _load_cv_meta(meta_path: str):
    """Загружаем label_map и строим idx->label. Кэш по meta_path."""
    with open(meta_path, "r") as f:
        meta_json = json.load(f)
    label_map = meta_json.get("label_map", meta_json)
    idx_to_label = {idx: lab for lab, idx in label_map.items()}
    num_classes = len(label_map)
    return label_map, idx_to_label, num_classes


@lru_cache(maxsize=4)
def load_cv_model(
    meta_path: str, model_path: str, device: Optional[str] = None
) -> Tuple["nn.Module", Dict[int, str]]:
    """
    Загрузка CV модели (ResNet50), натренированной в cv_model.ipynb.

    Кэшируется по (meta_path, model_path, device).
    """
    if torch is None or nn is None or tv_models is None:
        raise ImportError("torch/torchvision must be installed to load the CV model")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    _, idx_to_label, num_classes = _load_cv_meta(meta_path)

    model = create_cv_model_resnet50(num_classes=num_classes, device=device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, idx_to_label


def predict_cv(
    model: "nn.Module",
    img_tensor: "torch.Tensor",
    idx_to_label: Dict[int, str],
    device: Optional[str] = None,
    return_proba: bool = True,
) -> Dict[str, Any]:
    """
    Инференс CV-модели.

    Возвращает:
      - 'indices': список индексов классов
      - 'labels':  список строковых лейблов ('up'/'down'/'flat')
      - 'proba':   np.ndarray (N, C) с вероятностями (если return_proba=True)
    """
    if torch is None:
        raise ImportError("torch must be installed to perform CV inference")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()

    labels = [idx_to_label.get(int(i), str(i)) for i in pred_idx]

    result: Dict[str, Any] = {"indices": pred_idx.tolist(), "labels": labels}
    if return_proba:
        result["proba"] = probs_np
    return result


# =====================================================================
#                              RL часть
# =====================================================================

@lru_cache(maxsize=1)
def load_rl_agent(model_path: str, window_size: int, initial_balance: float) -> RLAgent:
    """Load and cache the PPO-based RL agent for inference."""
    return RLAgent(model_path=model_path, window_size=window_size, initial_balance=initial_balance)


def prepare_rl_state(
    df: pd.DataFrame,
    window_size: int,
    initial_balance: float,
    position: int = 0,
    equity: Optional[float] = None,
) -> Tuple[pd.DataFrame, PortfolioState]:
    """Фичи + состояние портфеля для RL."""
    features = build_feature_frame(df)
    if features.empty:
        raise ValueError("Cannot build RL features from an empty DataFrame")

    pointer = len(features) - 1
    state = PortfolioState(
        pointer=pointer,
        position=position,
        equity=equity if equity is not None else initial_balance,
    )
    return features, state


# =====================================================================
#                Форматирование блоков вывода ML / CV / RL
# =====================================================================

def format_ml_block(ml_preds: dict) -> str:
    if not ml_preds:
        return "ML: no predictions"

    proba = (
        ml_preds.get("last_proba")
        or ml_preds.get("proba")
        or ml_preds.get("probabilities")
    )
    label = ml_preds.get("last_label") or ml_preds.get("label")

    lines = ["ML predictions:"]
    if label is not None:
        lines.append(f"  label: {label}")
    if isinstance(proba, dict):
        lines.append("  probabilities:")
        for k, v in proba.items():
            lines.append(f"    {k}: {v:.4f}")
    return "\n".join(lines)


def format_cv_block(cv_preds: dict) -> str:
    if not cv_preds:
        return "CV: no predictions"

    proba = (
        cv_preds.get("last_proba")
        or cv_preds.get("proba")
        or cv_preds.get("probabilities")
    )
    label = cv_preds.get("last_label") or cv_preds.get("label")

    lines = ["CV predictions:"]
    if label is not None:
        lines.append(f"  label: {label}")
    if isinstance(proba, dict):
        lines.append("  probabilities:")
        for k, v in proba.items():
            lines.append(f"    {k}: {v:.4f}")
    return "\n".join(lines)


def format_rl_block(result: dict) -> str:
    if not result or "rl_action" not in result:
        return "RL: disabled"

    raw = result.get("rl_action")
    label = result.get("rl_action_label")
    size = result.get("rl_action_size")

    lines = ["RL action:"]
    lines.append(f"  raw_action: {raw}")
    if label is not None:
        lines.append(f"  label    : {label}")
    if size is not None:
        lines.append(f"  size_frac: {size:.4f}")
    return "\n".join(lines)


# =====================================================================
#     Утилита: перевод decision → JSON-пэйлоад для ордер-менеджера
# =====================================================================

def _to_jsonable(obj: Any) -> Any:
    """
    Внутренняя утилита: делает объект JSON-совместимым:
    - Timestamp/Datetime → isoformat строка,
    - остальные типы отдаёт как есть.
    """
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj


def decision_to_order_json(decision: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    """
    Преобразует decision-словарь из run_live_step в JSON-совместимый payload
    для ордер-менеджера.

    Если сделка НЕ должна открываться (нет 'open' или size<=0/dir=0), возвращает None.

    Формат ответа:
    {
      "ticker": "SBER",
      "side": "BUY" / "SELL",
      "size_frac": float,
      "tp": float | null,
      "sl": float | null,
      "price": float | null,
      "timestamp": str (ISO),
      "reason": str | null,
      "meta": {
          "decision_type": "open" / "no_trade_*",
          "raw_direction": -1/0/1
      }
    }
    """
    if not decision:
        return None

    decision_type = decision.get("type")
    if decision_type != "open":
        return None

    direction = int(decision.get("direction", 0) or 0)
    size_frac = float(decision.get("size_frac", 0.0) or 0.0)

    if direction == 0 or size_frac <= 0.0:
        return None

    side = "BUY" if direction > 0 else "SELL"

    payload: Dict[str, Any] = {
        "ticker": ticker,
        "side": side,
        "size_frac": float(size_frac),
        "tp": decision.get("tp"),
        "sl": decision.get("sl"),
        "price": decision.get("price"),
        "timestamp": _to_jsonable(decision.get("time")),
        "reason": decision.get("reason"),
        "meta": {
            "decision_type": decision_type,
            "raw_direction": direction,
        },
    }

    # Прогоняем через _to_jsonable то, что может быть не-примитивом
    for k in ("tp", "sl", "price"):
        if payload[k] is not None:
            payload[k] = float(payload[k])

    return payload


# =====================================================================
#                     Быстрый лайв-шаг: run_live_step
# =====================================================================

def run_live_step(
    token: str,
    ticker: str = "SBER",
    lookback_cv: int = 60,
    rl_window: int = 32,
    initial_balance: float = 100_000.0,
    ml_model_path: str = "app/best_ssm_model.pth",
    cv_model_path: str = "app/best_cv_model.pth",
    meta_path: str = "app/label_map.json",
    rl_model_path: str = "models/rl/checkpoints/ppo_trading_agent.zip",
    days_for_fetch: int = 10,
    tz=DEFAULT_TZ,
    interval=DEFAULT_TIMEFRAME,
    ml_conf_thr: float = ML_PROBA_THR_LIVE,
    cv_conf_thr: float = CV_PROBA_THR_LIVE,
) -> Dict[str, Any]:
    """
    Один быстрый шаг инференса по последней свече (без бэктеста).

    Возвращает:
    {
      "df_candles": df,
      "ml_summary": {...},
      "cv_summary": {...},
      "rl_summary": {...},
      "decision": {
          "type": "open" / "no_trade_*" / "warmup_not_enough_data" / "ml_no_output",
          "reason": str,
          "direction": -1/0/1,
          "size_frac": float,
          "tp": float | None,
          "sl": float | None,
      }
    }
    """
    if not token:
        raise RuntimeError("Tinkoff API token is empty or None")

    # 1) свечи
    df = fetch_tinkoff_candles(
        token=token,
        ticker=ticker,
        days=days_for_fetch,
        interval=interval,
        tz=tz,
    ).sort_values("time").reset_index(drop=True)

    if df.empty:
        return {
            "df_candles": df,
            "ml_summary": {},
            "cv_summary": {},
            "rl_summary": {},
            "decision": {
                "type": "no_data",
                "reason": "No candles returned from API",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            },
        }

    # 2) ATR
    df["atr_14"] = compute_atr_wilder(df, n=14)

    # warmup по длине
    warmup = max(lookback_cv, rl_window, SSM_SEQ_LEN + 20)
    if len(df) < warmup:
        return {
            "df_candles": df,
            "ml_summary": {},
            "cv_summary": {},
            "rl_summary": {},
            "decision": {
                "type": "warmup_not_enough_data",
                "reason": f"Need at least {warmup} candles, have {len(df)}",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            },
        }

    hist = df.copy()
    last_row = hist.iloc[-1]
    ts = last_row["time"]
    close_price = float(last_row["close"])

    # ---------------- ML (SSM) ----------------
    ml_res = predict_ml_ssm(hist, model_path=ml_model_path, return_proba=True)
    if len(ml_res["labels"]) == 0:
        return {
            "df_candles": df,
            "ml_summary": {},
            "cv_summary": {},
            "rl_summary": {},
            "decision": {
                "type": "ml_no_output",
                "reason": "SSM model returned no predictions",
                "direction": 0,
                "size_frac": 0.0,
                "tp": None,
                "sl": None,
            },
        }

    ml_label = ml_res["labels"][-1]
    ml_probs = ml_res["proba"][-1]
    ml_idx = ml_res["indices"][-1]
    ml_conf = float(ml_probs[ml_idx])

    ml_full_proba = {
        "down": float(ml_probs[0]),
        "flat": float(ml_probs[1]),
        "up": float(ml_probs[2]),
    }
    ml_summary = {
        "label": ml_label,
        "proba": ml_full_proba,
    }

    if DEBUG:
        print("LIVE ML:", ml_label, ml_conf, ml_full_proba)

    # ---------------- CV ----------------
    cv_model, idx_to_label = load_cv_model(meta_path, cv_model_path)

    cv_df = add_cv_features(hist)
    sub = cv_df.iloc[-lookback_cv:].copy()
    sub["time"] = pd.to_datetime(sub["time"])
    sub = sub.set_index("time")

    img = render_candle_image(sub, img_size=(8, 4), dpi=100, use_jpg=True)
    img_tensor = prepare_cv_tensor(img)

    cv_res = predict_cv(cv_model, img_tensor, idx_to_label, return_proba=True)
    cv_label = cv_res["labels"][-1]
    cv_probs = cv_res["proba"][-1]
    cv_idx = cv_res["indices"][-1]
    cv_conf = float(cv_probs[cv_idx])

    cv_full_proba: Dict[str, float] = {}
    for idx, p in enumerate(cv_probs):
        lab = idx_to_label.get(int(idx), str(idx))
        cv_full_proba[lab] = float(p)

    cv_summary = {
        "label": cv_label,
        "proba": cv_full_proba,
    }

    if DEBUG:
        print("LIVE CV:", cv_label, cv_conf, cv_full_proba)

    # ---------------- RL ----------------
    rl_agent = load_rl_agent(rl_model_path, rl_window, initial_balance)
    rl_features, rl_state = prepare_rl_state(
        hist,
        window_size=rl_window,
        initial_balance=initial_balance,
        position=0,
        equity=initial_balance,  # в live-режиме статичное значение
    )
    raw_pos, rl_action_label, rl_size = rl_agent.get_action(rl_features, rl_state)

    rl_summary = {
        "rl_action": float(raw_pos),
        "rl_action_label": rl_action_label,
        "rl_action_size": float(rl_size),
    }

    if DEBUG:
        print("LIVE RL:", raw_pos, rl_action_label, rl_size)

    # ================= Базовая логика ML+CV+RL =================

    # flat → не торгуем
    if ml_label == "flat" or cv_label == "flat":
        decision = {
            "type": "no_trade_flat",
            "reason": f"ML={ml_label}, CV={cv_label} (flat present)",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    # разъезд ML/CV → не торгуем
    if ml_label != cv_label:
        decision = {
            "type": "no_trade_disagree",
            "reason": f"ML={ml_label}, CV={cv_label} disagree",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    # низкая уверенность → не торгуем
    if ml_conf < ml_conf_thr or cv_conf < cv_conf_thr:
        decision = {
            "type": "no_trade_low_conf",
            "reason": f"low confidence: ml_conf={ml_conf:.4f}, cv_conf={cv_conf:.4f}",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    # направление по ML/CV (они уже совпадают)
    direction = 1 if ml_label == "up" else -1

    # RL: нулевая/отрицательная величина → не торгуем
    if rl_size <= 0.0:
        decision = {
            "type": "no_trade_rl_flat",
            "reason": f"RL size <= 0: raw_action={raw_pos}, size={rl_size}",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    rl_direction = 1 if raw_pos > 0 else -1
    if rl_direction != direction:
        decision = {
            "type": "no_trade_rl_disagree",
            "reason": f"RL direction ({rl_direction}) != ML/CV direction ({direction})",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    # ---------------- ATR и базовый TP/SL ----------------
    atr_last = float(hist["atr_14"].iloc[-1])
    if not np.isfinite(atr_last) or atr_last <= 0:
        decision = {
            "type": "no_trade_no_atr",
            "reason": f"Invalid ATR value: {atr_last}",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    if direction > 0:
        tp_price = close_price + TP_ATR_K * atr_last
        sl_price = close_price - SL_ATR_K * atr_last
    else:
        tp_price = close_price - TP_ATR_K * atr_last
        sl_price = close_price + SL_ATR_K * atr_last

    # ---------------- Агрегированные proba up/down ----------------
    proba_up_vals = [ml_full_proba.get("up", np.nan)]
    proba_down_vals = [ml_full_proba.get("down", np.nan)]

    cv_up = cv_full_proba.get("up", np.nan)
    cv_down = cv_full_proba.get("down", np.nan)
    if np.isfinite(cv_up):
        proba_up_vals.append(cv_up)
    if np.isfinite(cv_down):
        proba_down_vals.append(cv_down)

    proba_up = float(np.nanmean(proba_up_vals)) if len(proba_up_vals) > 0 else np.nan
    proba_down = (
        float(np.nanmean(proba_down_vals)) if len(proba_down_vals) > 0 else np.nan
    )

    # ---------------- Продвинутый RM в отдельной функции ----------------
    rm_decision, size_frac = apply_risk_overlays(
        hist=hist,
        direction=direction,
        close_price=close_price,
        tp_price=tp_price,
        sl_price=sl_price,
        proba_up=proba_up,
        proba_down=proba_down,
        rl_size=rl_size,
    )

    if rm_decision is not None:
        # no_trade_* из риск-оверлеев
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": rm_decision,
        }

    # финальная защита (на случай странных чисел)
    if size_frac <= 0.0:
        decision = {
            "type": "no_trade_zero_size",
            "reason": "size_frac <= 0 после risk overlays",
            "direction": 0,
            "size_frac": 0.0,
            "tp": None,
            "sl": None,
        }
        return {
            "df_candles": df,
            "ml_summary": ml_summary,
            "cv_summary": cv_summary,
            "rl_summary": rl_summary,
            "decision": decision,
        }

    # ---------------- Финальное решение: open ----------------
    decision = {
        "type": "open",
        "reason": (
            f"ML={ml_label}, CV={cv_label}, RL={rl_action_label}, "
            f"prob_up={proba_up:.3f}, prob_down={proba_down:.3f}, size_frac={size_frac:.3f}"
        ),
        "direction": direction,
        "size_frac": size_frac,
        "tp": float(tp_price),
        "sl": float(sl_price),
        "time": ts,
        "price": close_price,
    }

    return {
        "df_candles": df,
        "ml_summary": ml_summary,
        "cv_summary": cv_summary,
        "rl_summary": rl_summary,
        "decision": decision,
    }


# =====================================================================
#                     Один запуск + главный цикл
# =====================================================================

def run_once() -> None:
    """
    Один запуск live-инференса:

      - run_live_step,
      - печать последней свечи,
      - печать предсказаний ML/CV/RL,
      - печать решения,
      - печать JSON-пэйлоада для ордер-менеджера (если есть сделка).
    """
    if not DEFAULT_TOKEN:
        raise RuntimeError("TINKOFF_TOKEN is not set in .env")

    result = run_live_step(
        token=DEFAULT_TOKEN,
        ticker=TICKER,
        lookback_cv=LOOKBACK,
        rl_window=WINDOW,
        initial_balance=INITIAL_BALANCE,
        ml_model_path=ML_MODEL_PATH,
        cv_model_path=CV_MODEL_PATH,
        meta_path=META_CV_PATH,
        rl_model_path=RL_MODEL_PATH,
        days_for_fetch=DAYS_FOR_FETCH,
        tz=DEFAULT_TZ,
        interval=DEFAULT_TIMEFRAME
    )

    df_candles: pd.DataFrame | None = result.get("df_candles")
    ml_summary: dict | None = result.get("ml_summary") or {}
    cv_summary: dict | None = result.get("cv_summary") or {}
    rl_summary: dict | None = result.get("rl_summary") or {}
    decision: dict | None = result.get("decision") or {}

    print("-" * 80)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Live inference result:")

    if df_candles is not None and not df_candles.empty:
        last_row = df_candles.iloc[-1]
        ts = last_row["time"]
        o = float(last_row["open"])
        h = float(last_row["high"])
        l = float(last_row["low"])
        c = float(last_row["close"])
        v = float(last_row.get("volume", 0.0))
        print("Last candle (from API):")
        print(f"  time   : {ts}")
        print(f"  O/H/L/C: {o:.4f} / {h:.4f} / {l:.4f} / {c:.4f}")
        print(f"  volume : {v}")
    else:
        print("No candles returned in df_candles.")

    print()
    print(format_ml_block(ml_summary))
    print()
    print(format_cv_block(cv_summary))
    print()
    print(format_rl_block(rl_summary))
    print()

    print("Decision:")
    if not decision:
        print("  no decision object")
    else:
        print(f"  type   : {decision.get('type')}")
        print(f"  reason : {decision.get('reason')}")
        print(f"  dir    : {decision.get('direction')}")
        print(f"  size   : {decision.get('size_frac')}")
        print(f"  tp/sl  : {decision.get('tp')} / {decision.get('sl')}")
        if "time" in decision:
            print(f"  at     : {decision.get('time')} (price={decision.get('price')})")

    print()

    # Пример: формируем JSON-пэйлоад для ордер-менеджера
    order_payload = decision_to_order_json(decision, ticker=TICKER)
    if order_payload is not None:
        print("Order payload (JSON-ready):")
        print(json.dumps(order_payload, ensure_ascii=False, default=_to_jsonable, indent=2))
    else:
        print("Order payload: None (no open trade).")

    print("-" * 80)
    print()


def main():
    """Главный цикл: запускать инференс раз в 5 минут бесконечно."""
    while True:
        try:
            run_once()
        except Exception as e:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("=" * 80)
            print(f"[{now_str}] ERROR during inference: {e}")
            print("=" * 80)
            print()
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
