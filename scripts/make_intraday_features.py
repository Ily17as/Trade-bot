import pandas as pd, numpy as np, glob, argparse, os
from scripts.paths import FEAT

def add_session_cols(df):
    dt=pd.to_datetime(df["time"])
    df["date"]=dt.dt.date
    df["bar_id"]=df.groupby("date").cumcount()
    return df

def vwap20(df):
    tp=(df["high"]+df["low"]+df["close"])/3
    vol=df["volume"].replace(0,np.nan)
    return (tp*vol).rolling(20,min_periods=5).sum()/vol.rolling(20,min_periods=5).sum()

def build_features(parts_glob: str) -> pd.DataFrame:
    parts=glob.glob(parts_glob); assert parts, "raw parquet not found"
    df=(pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
          .drop_duplicates("time").sort_values("time"))
    px=df["close"].astype(float)
    df["ret_1"]=np.log(px).diff()
    df["ret_5"]=np.log(px).diff(5)
    df["rng_1"]=(df["high"]-df["low"]).div(px.shift())
    df["vol_20"]=df["ret_1"].rolling(20,min_periods=10).std()
    df["vwap_20"]=vwap20(df)
    df["dist_vwap"]=(px-df["vwap_20"]).div(px)
    df=add_session_cols(df)
    df["is_open_30"]=(df["bar_id"]<30).astype(int)
    df["is_close_30"]=df.groupby("date")["bar_id"].transform(lambda s:(s.max()-s)<30).astype(int)
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source", default="tinkoff")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--tf", default="1m")
    args=ap.parse_args()
    parts=f"data/raw/{args.source}/{args.ticker}/{args.tf}/year=*/month=*/part.parquet"
    fe=build_features(parts)
    out=FEAT/f"ml/{args.ticker}/{args.tf}/features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    fe.to_parquet(out, index=False); print(out)

if __name__=="__main__": main()
