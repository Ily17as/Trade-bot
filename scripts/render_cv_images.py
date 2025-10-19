# scripts/render_cv_images.py
import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------- utils -------------------------

def _to_float(p):
    return float(p)

def _vol_transform(v: pd.Series, mode: str = "raw", roll: int = 20) -> pd.Series:
    v = pd.Series(v).astype(float)
    if mode == "raw":
        return v
    if mode == "zscore":
        m = v.rolling(roll, min_periods=max(5, roll // 2)).mean()
        s = v.rolling(roll, min_periods=max(5, roll // 2)).std()
        return (v - m) / s.replace(0.0, np.nan)
    if mode == "rel":
        m = v.rolling(roll, min_periods=max(5, roll // 2)).mean()
        return v / m.replace(0.0, np.nan)
    return v

def _compute_label(df: pd.DataFrame, i: int, size: int, horizon: int,
                   label_thr: float) -> Tuple[str, float]:
    """Возвращает (label, ret). Если будущего окна нет -> ('none', nan)."""
    last_idx = i + size - 1
    fut_idx = last_idx + horizon
    if fut_idx >= len(df):
        return "none", np.nan
    c0 = float(df["close"].iloc[last_idx])
    c1 = float(df["close"].iloc[fut_idx])
    ret = np.log(c1) - np.log(c0)
    if ret > label_thr:
        return "up", ret
    if ret < -label_thr:
        return "down", ret
    return "flat", ret

def _fig_ax(height_px: int, width_px: int, n_panels: int) -> Tuple[plt.Figure, List[plt.Axes]]:
    # Делаем точный размер PNG: dpi = height_px, figsize = (w/h, 1)
    dpi = height_px
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi),
                     dpi=dpi,
                     constrained_layout=False
                     )
    if n_panels == 1:
        gs = GridSpec(1, 1, figure=fig, hspace=0.0)
        axs = [fig.add_subplot(gs[0, 0])]
    elif n_panels == 2:
        gs = GridSpec(4, 1, figure=fig, hspace=0.0, height_ratios=[3, 0, 1, 0])
        axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[2, 0])]
    else:  # 3 панели: цена, объём, extra_ch
        gs = GridSpec(6, 1, figure=fig, hspace=0.0, height_ratios=[3, 0, 1.2, 0, 1.2, 0])
        axs = [fig.add_subplot(gs[0, 0]),
               fig.add_subplot(gs[2, 0]),
               fig.add_subplot(gs[4, 0])]
    for ax in axs:
        ax.axis("off")
    return fig, axs

def _apply_ylim(ax: plt.Axes, ymin: float, ymax: float,
                fixed_ylim, ylim_pad: float):
    if isinstance(fixed_ylim, tuple) and len(fixed_ylim) == 2:
        ax.set_ylim(fixed_ylim)
        return
    if fixed_ylim == "window":
        pad = (ymax - ymin) * float(max(0.0, ylim_pad))
        ax.set_ylim(ymin - pad, ymax + pad)

# ------------------------- core rendering -------------------------

def _draw_price(ax: plt.Axes, win: pd.DataFrame, mode: str):
    t = np.arange(len(win))
    o = win["open"].to_numpy(dtype=float)
    h = win["high"].to_numpy(dtype=float)
    l = win["low"].to_numpy(dtype=float)
    c = win["close"].to_numpy(dtype=float)

    if mode == "candle":
        # упрощённые свечи: вертикальные "фитили" + прямоугольник тела
        for i in range(len(win)):
            ax.vlines(i, l[i], h[i], linewidth=1)
            body_y0, body_y1 = sorted([o[i], c[i]])
            ax.vlines([i - 0.3, i + 0.3], [body_y0, body_y0], [body_y1, body_y1], linewidth=2)
            ax.hlines([body_y0, body_y1], i - 0.3, i + 0.3, linewidth=2)
        return

    if mode == "close":
        ax.plot(t, c, linewidth=1)
        return

    if mode == "close+hl_wick":
        for i in range(len(win)):
            ax.vlines(i, l[i], h[i], linewidth=1)  # тени HL
        ax.plot(t, c, linewidth=1)  # линия Close
        return

    # fallback
    ax.plot(t, c, linewidth=1)

def _draw_volume(ax: plt.Axes, v: pd.Series):
    t = np.arange(len(v))
    ax.bar(t, v.to_numpy(dtype=float), width=0.8, linewidth=0)

def _draw_extra(ax: plt.Axes, s: pd.Series):
    t = np.arange(len(s))
    ax.plot(t, s.to_numpy(dtype=float), linewidth=1)

# ------------------------- public API -------------------------


def render_window(win_df: pd.DataFrame, out_path: Path,
                  show_axes: bool = False,
                  fixed_ylim=None,
                  show_volume: bool = False,
                  vol_mode: str = "raw",
                  vol_roll: int = 20,
                  fixed_vol_ylim: Optional[Tuple[float, float]] = None,
                  ohlc_mode: str = "close+hl_wick",
                  ylim_pad: float = 0.0,
                  extra_ch_series: Optional[pd.Series] = None):
    """
    Рисует окно в PNG. Цена сверху, затем объём, затем extra_ch (если есть).
    """
    n_panels = 1 + int(show_volume) + int(extra_ch_series is not None)
    fig, axs = _fig_ax(height_px=64, width_px=64, n_panels=n_panels)

    # --- Price panel ---
    ax_price = axs[0]
    _draw_price(ax_price, win_df, ohlc_mode)
    ymin, ymax = float(win_df["low"].min()), float(win_df["high"].max())
    _apply_ylim(ax_price, ymin, ymax, fixed_ylim, ylim_pad)
    if show_axes:
        ax_price.axis("on")

    # --- Volume panel ---
    if show_volume:
        ax_v = axs[1]
        v = _vol_transform(win_df["volume"], vol_mode, vol_roll)
        vmin, vmax = (fixed_vol_ylim if fixed_vol_ylim is not None
                      else (float(np.nanmin(v)), float(np.nanmax(v))))
        v = v.clip(vmin, vmax)
        _draw_volume(ax_v, v)
        if show_axes:
            ax_v.axis("on")

    # --- Extra channel panel ---
    if extra_ch_series is not None:
        ax_e = axs[-1]
        s = pd.Series(extra_ch_series).astype(float)
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        if smin == smax:
            smax = smin + 1e-9
        ax_e.set_ylim(smin, smax)
        _draw_extra(ax_e, s)
        if show_axes:
            ax_e.axis("on")

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches=None, pad_inches=0)
    plt.close(fig)

def render_with_manifest(in_parquet: str, ticker: str, tf: str,
                         size: int = 64, step: int = 16, horizon: int = 5,
                         show_axes: bool = False,
                         fixed_ylim=None,
                         show_volume: bool = False,
                         vol_mode: str = "raw",
                         vol_roll: int = 20,
                         fixed_vol_ylim: Optional[Tuple[float, float]] = None,
                         manifest_csv: Optional[str] = None,
                         ylim_pad: float = 0.0,
                         ohlc_mode: str = "close+hl_wick",
                         extra_ch: Optional[str] = None,
                         seed: Optional[int] = None,
                         label_col: Optional[str] = None,
                         label_thr: float = 0.004) -> List[dict]:
    """
    Генерирует PNG и CSV-манифест.
    - fixed_ylim="window" — автодиапазон по окну с паддингом ylim_pad.
    - ohlc_mode: "candle" | "close" | "close+hl_wick".
    - vol_mode: "raw" | "zscore" | "rel".
    - extra_ch: имя столбца в df, который будет выведен в отдельной нижней панели.
    - label_col: если None, пытаемся использовать f"y_tb_{horizon}", иначе считаем по label_thr.
    """
    if seed is not None:
        np.random.seed(int(seed))

    df = pd.read_parquet(in_parquet).sort_values("time").reset_index(drop=True)

    # колонка метки
    lbl_col = label_col or (f"y_tb_{horizon}" if f"y_tb_{horizon}" in df.columns else None)

    base = Path(f"data/cv/images/{ticker}/{tf}/w{size}_s{step}")
    base.mkdir(parents=True, exist_ok=True)

    rows, idx = [], 0
    n = len(df)
    for i in range(0, n - size + 1, step):
        win = df.iloc[i:i + size].copy()

        # --- label & ret ---
        if lbl_col and lbl_col in df.columns:
            val = df[lbl_col].iloc[i + size - 1]
            if pd.isna(val):
                label, ret = "none", np.nan
            else:
                if isinstance(val, (int, np.integer)) and val in (-1, 0, 1):
                    label = {1: "up", 0: "flat", -1: "down"}[int(val)]
                else:
                    label = str(val)
                # контрольный ret для лога
                label, ret = label, np.nan
        else:
            label, ret = _compute_label(df, i, size, horizon, label_thr)

        # путь сохранения
        outdir = base / label
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"img_{idx:07d}.png"

        # --- extra channel series (если есть) ---
        extra_series = None
        if extra_ch and extra_ch in win.columns:
            extra_series = win[extra_ch]

        # --- fixed ylim обработка ---
        ylims = None
        if fixed_ylim == "window":
            ymin, ymax = float(win["low"].min()), float(win["high"].max())
            ylims = (ymin, ymax)
        elif isinstance(fixed_ylim, tuple):
            ylims = fixed_ylim

        # рендер
        render_window(
            win_df=win,
            out_path=out_path,
            show_axes=show_axes,
            fixed_ylim=ylims,
            show_volume=show_volume,
            vol_mode=vol_mode,
            vol_roll=vol_roll,
            fixed_vol_ylim=fixed_vol_ylim,
            ohlc_mode=ohlc_mode,
            ylim_pad=ylim_pad,
            extra_ch_series=extra_series,
        )

        # статистика для манифеста
        v = _vol_transform(win["volume"], vol_mode, vol_roll) if show_volume else win["volume"]
        vmin = float(np.nanmin(v)) if len(v) else None
        vmax = float(np.nanmax(v)) if len(v) else None
        ymin = float(win["low"].min())
        ymax = float(win["high"].max())

        rows.append({
            "path": str(out_path),
            "ticker": ticker, "tf": tf,
            "t_start": str(win["time"].iloc[0]),
            "t_end": str(win["time"].iloc[-1]),
            "bars": int(len(win)),
            "ymin": ymin, "ymax": ymax,
            "vol_mode": vol_mode,
            "vmin": vmin, "vmax": vmax,
            "label": label,
            "ret_fwd": ret,
            "horizon": int(horizon),
            "extra_ch": extra_ch if extra_ch else "",
        })
        idx += 1

    if manifest_csv:
        Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(manifest_csv, index=False)

    return rows












# # добавьте в начало файла
# import numpy as np, pandas as pd, os
# import mplfinance as mpf
# from pathlib import Path
#
# def _vol_transform(v, mode="raw", roll=20):
#     v = pd.Series(v).astype(float)
#     if mode == "raw":
#         return v
#     if mode == "zscore":
#         m = v.rolling(roll, min_periods=max(5, roll//2)).mean()
#         s = v.rolling(roll, min_periods=max(5, roll//2)).std()
#         return (v - m) / s.replace(0, np.nan)
#     if mode == "rel":  # relative volume = v / rolling_mean
#         m = v.rolling(roll, min_periods=max(5, roll//2)).mean()
#         return v / m.replace(0, np.nan)
#     return v
#
# def render_window(win_df, out_path,
#                   show_axes=False,
#                   fixed_ylim=None,
#                   show_volume=False,
#                   vol_mode="raw",
#                   vol_roll=20,
#                   fixed_vol_ylim=None):
#     dfp = win_df.set_index("time")[["open","high","low","close","volume"]]
#     addp = None
#     plot_kwargs = dict(type='candle', volume=False, style="default",
#                        axisoff=not show_axes,
#                        savefig=dict(fname=str(out_path), dpi=128, bbox_inches="tight", pad_inches=0))
#     # верхняя панель (цены)
#     if fixed_ylim is not None:
#         plot_kwargs["ylim"] = fixed_ylim
#
#     # нижняя панель (объёмы)
#     if show_volume:
#         v = _vol_transform(dfp["volume"], vol_mode, vol_roll)
#         vmin, vmax = (fixed_vol_ylim if fixed_vol_ylim is not None
#                       else (float(np.nanmin(v)), float(np.nanmax(v))))
#         addp = [mpf.make_addplot(v, type="bar", panel=1, ylim=(vmin, vmax))]
#         plot_kwargs["addplot"] = addp
#         plot_kwargs["panel_ratios"] = (3, 1)
#
#     mpf.plot(dfp[["open","high","low","close","volume"]], **plot_kwargs)
#     return
#
# def render_with_manifest(in_parquet, ticker, tf,
#                          size=64, step=16, horizon=5,
#                          show_axes=False,
#                          fixed_ylim=None,
#                          show_volume=False,
#                          vol_mode="raw",
#                          vol_roll=20,
#                          fixed_vol_ylim=None,
#                          manifest_csv=None):
#     df = pd.read_parquet(in_parquet).sort_values("time")
#     base = Path(f"data/cv/images/{ticker}/{tf}/win{size}_step{step}")
#     base.mkdir(parents=True, exist_ok=True)
#     rows, idx = [], 0
#     for i in range(0, len(df)-size+1, step):
#         win = df.iloc[i:i+size].copy()
#         # простая метка по будущему ретурну
#         t_last_idx = i + size - 1
#         t_fut_idx = t_last_idx + horizon
#         if t_fut_idx < len(df):
#             ret = np.log(df["close"].iloc[t_fut_idx]) - np.log(df["close"].iloc[t_last_idx])
#         else:
#             ret = np.nan  # нет будущего окна -> пометьте как None/пропустите
#         thr = 0.004
#         lab = ("up" if ret > thr else ("down" if ret < -thr else "flat")) if pd.notna(ret) else "none"
#
#         ymin, ymax = float(win["low"].min()), float(win["high"].max())
#         outdir = base / lab
#         outdir.mkdir(parents=True, exist_ok=True)
#         out_path = outdir / f"img_{idx:07d}.png"
#
#         render_window(win, out_path,
#                       show_axes=show_axes,
#                       fixed_ylim=(ymin, ymax) if fixed_ylim == "window" else fixed_ylim,
#                       show_volume=show_volume,
#                       vol_mode=vol_mode,
#                       vol_roll=vol_roll,
#                       fixed_vol_ylim=fixed_vol_ylim)
#
#         v = _vol_transform(win["volume"], vol_mode, vol_roll) if show_volume else win["volume"]
#         vmin = float(np.nanmin(v)) if len(v) else None
#         vmax = float(np.nanmax(v)) if len(v) else None
#
#         rows.append({
#             "path": str(out_path),
#             "ticker": ticker, "tf": tf,
#             "t_start": str(win["time"].iloc[0]),
#             "t_end": str(win["time"].iloc[-1]),
#             "bars": int(len(win)),
#             "ymin": ymin, "ymax": ymax,
#             "vol_mode": vol_mode,
#             "vmin": vmin, "vmax": vmax,
#             "label": lab
#         })
#         idx += 1
#     if manifest_csv:
#         Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
#         pd.DataFrame(rows).to_csv(manifest_csv, index=False)
#     return rows
