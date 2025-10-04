import pandas as pd, os
import mplfinance as mpf
from pathlib import Path

def windows(df, size=64, step=16):
    for i in range(0, len(df)-size+1, step):
        yield df.iloc[i:i+size]

def label_next(win_df, horizon=5, thr=0.004):
    last=win_df["close"].iloc[-1]
    j=min(len(win_df)-1+horizon, len(win_df)-1)
    ret=(win_df["close"].iloc[j]/last)-1
    return "up" if ret>thr else ("down" if ret<-thr else "flat")

def render(in_parquet, ticker, tf, size=64, step=16, horizon=5):
    df=pd.read_parquet(in_parquet).sort_values("time")
    base=Path(f"data/cv/images/{ticker}/{tf}/win{size}_step{step}"); base.mkdir(parents=True, exist_ok=True)
    idx=0
    for win in windows(df,size,step):
        lab=label_next(win,horizon)
        dfi=win.set_index("time")[["open","high","low","close","volume"]]
        out=(base/lab); out.mkdir(parents=True, exist_ok=True)
        mpf.plot(dfi, type="candle", volume=False, axisoff=True,
                 savefig=dict(fname=str(out/f"img_{idx:07d}.png"), dpi=128, bbox_inches="tight", pad_inches=0))
        idx+=1
