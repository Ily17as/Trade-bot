import pandas as pd, glob

def qc_summary(glob_path):
    parts=glob.glob(glob_path);
    if not parts: return pd.DataFrame()
    df=(pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("time").sort_values("time"))
    df["gap"]=pd.to_datetime(df["time"]).diff()
    return pd.DataFrame([{
        "rows":len(df),
        "zero_price":int(((df[["open","high","low","close"]]<=0).any(axis=1)).sum()),
        "bad_hl":int((df["high"]<df["low"]).sum()),
        "zero_vol":int((df["volume"]<=0).sum()),
        "max_gap":str(df["gap"].max())
    }])
