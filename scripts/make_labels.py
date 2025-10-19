import pandas as pd, numpy as np, argparse

def next_k_return(df, k=5):
    c = df["close"].astype(float)
    f = c.shift(-k)
    df[f"ret_fwd_{k}"] = np.log(f) - np.log(c)
    df.loc[df.index[-k:], f"ret_fwd_{k}"] = np.nan
    return df

def triple_barrier(df, k=20, up=0.004, dn=0.004):
    c = df["close"].astype(float).to_numpy(); n = len(c)
    y = np.full(n, np.nan)  # маскируем хвост по умолчанию
    for i in range(0, n - k):           # хвост не трогаем
        up_lvl, dn_lvl = c[i]*(1+up), c[i]*(1-dn)
        path = c[i+1:i+k+1]
        # "первое касание" вместо "оба -> flat"
        hit_up = np.where(path >= up_lvl)[0]
        hit_dn = np.where(path <= dn_lvl)[0]
        t_up = hit_up[0] if hit_up.size else np.inf
        t_dn = hit_dn[0] if hit_dn.size else np.inf
        y[i] = 1 if t_up < t_dn else (-1 if t_dn < t_up else 0)
    df[f"y_tb_{k}"] = y
    return df

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--k", type=int, default=20)
    args=ap.parse_args()
    df=pd.read_parquet(args.in_parquet).sort_values("time")
    df=next_k_return(df, k=5)
    df=triple_barrier(df, k=args.k, up=0.004, dn=0.004)
    df.to_parquet(args.out_parquet, index=False); print(args.out_parquet)

if __name__=="__main__": main()
