# добавьте в начало файла
import numpy as np, pandas as pd, os
import mplfinance as mpf
from pathlib import Path

def _vol_transform(v, mode="raw", roll=20):
    v = pd.Series(v).astype(float)
    if mode == "raw":
        return v
    if mode == "zscore":
        m = v.rolling(roll, min_periods=max(5, roll//2)).mean()
        s = v.rolling(roll, min_periods=max(5, roll//2)).std()
        return (v - m) / s.replace(0, np.nan)
    if mode == "rel":  # relative volume = v / rolling_mean
        m = v.rolling(roll, min_periods=max(5, roll//2)).mean()
        return v / m.replace(0, np.nan)
    return v

def render_window(win_df, out_path,
                  show_axes=False,
                  fixed_ylim=None,
                  show_volume=False,
                  vol_mode="raw",
                  vol_roll=20,
                  fixed_vol_ylim=None):
    dfp = win_df.set_index("time")[["open","high","low","close","volume"]]
    addp = None
    plot_kwargs = dict(type='candle', volume=False, style="default",
                       axisoff=not show_axes,
                       savefig=dict(fname=str(out_path), dpi=128, bbox_inches="tight", pad_inches=0))
    # верхняя панель (цены)
    if fixed_ylim is not None:
        plot_kwargs["ylim"] = fixed_ylim

    # нижняя панель (объёмы)
    if show_volume:
        v = _vol_transform(dfp["volume"], vol_mode, vol_roll)
        vmin, vmax = (fixed_vol_ylim if fixed_vol_ylim is not None
                      else (float(np.nanmin(v)), float(np.nanmax(v))))
        addp = [mpf.make_addplot(v, type="bar", panel=1, ylim=(vmin, vmax))]
        plot_kwargs["addplot"] = addp
        plot_kwargs["panel_ratios"] = (3, 1)

    mpf.plot(dfp[["open","high","low","close","volume"]], **plot_kwargs)
    return

def render_with_manifest(in_parquet, ticker, tf,
                         size=64, step=16, horizon=5,
                         show_axes=False,
                         fixed_ylim=None,
                         show_volume=False,
                         vol_mode="raw",
                         vol_roll=20,
                         fixed_vol_ylim=None,
                         manifest_csv=None):
    df = pd.read_parquet(in_parquet).sort_values("time")
    base = Path(f"data/cv/images/{ticker}/{tf}/win{size}_step{step}")
    base.mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    for i in range(0, len(df)-size+1, step):
        win = df.iloc[i:i+size].copy()
        # простая метка по будущему ретурну
        future_idx = min(len(win)-1+5, len(win)-1)
        ret = np.log(win["close"].iloc[future_idx]) - np.log(win["close"].iloc[-1])
        lab = "up" if ret > 0.004 else ("down" if ret < -0.004 else "flat")

        ymin, ymax = float(win["low"].min()), float(win["high"].max())
        outdir = base / lab
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"img_{idx:07d}.png"

        render_window(win, out_path,
                      show_axes=show_axes,
                      fixed_ylim=(ymin, ymax) if fixed_ylim == "window" else fixed_ylim,
                      show_volume=show_volume,
                      vol_mode=vol_mode,
                      vol_roll=vol_roll,
                      fixed_vol_ylim=fixed_vol_ylim)

        v = _vol_transform(win["volume"], vol_mode, vol_roll) if show_volume else win["volume"]
        vmin = float(np.nanmin(v)) if len(v) else None
        vmax = float(np.nanmax(v)) if len(v) else None

        rows.append({
            "path": str(out_path),
            "ticker": ticker, "tf": tf,
            "t_start": str(win["time"].iloc[0]),
            "t_end": str(win["time"].iloc[-1]),
            "bars": int(len(win)),
            "ymin": ymin, "ymax": ymax,
            "vol_mode": vol_mode,
            "vmin": vmin, "vmax": vmax,
            "label": lab
        })
        idx += 1
    if manifest_csv:
        Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(manifest_csv, index=False)
    return rows
