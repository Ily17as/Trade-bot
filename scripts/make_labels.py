import pandas as pd, numpy as np, argparse

def next_k_return(df,k=5):
    c=df["close"].to_numpy(); f=np.roll(c,-k)
    df[f"ret_fwd_{k}"]=np.log(f)-np.log(c); return df

def triple_barrier(df,k=20,up=0.004,dn=0.004):
    c=df["close"].to_numpy(); n=len(c); y=np.zeros(n,dtype=int)
    for i in range(n):
        j=min(i+k,n-1)
        seg=c[i+1:j+1]
        up_hit=(seg>=c[i]*(1+up)).any() if seg.size else False
        dn_hit=(seg<=c[i]*(1-dn)).any() if seg.size else False
        y[i]=1 if up_hit and not dn_hit else (-1 if dn_hit and not up_hit else 0)
    df[f"y_tb_{k}"]=y; return df

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
