# make_signals_from_inference.py
import os, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timezone
from inference import infer_ml, infer_cv, infer_math_from_prices  # из вашего inference.py

TICKER="SBER"; TF="5m"; H=36; THRESH_PCT=0.02
FEAT_TODAY = f"../data/features/ml/{TICKER}/{TF}/features_today.parquet"
CV_MANIFEST= f"../data/cv/images/{TICKER}/{TF}/today_manifest.csv"
ML_PKL     = f"../models/{TICKER}_{TF}_lgbm.pkl"
CV_PT      = f"../models/cv_resnet18_state.pt"
SIGNALS_CSV= f"../data/signals/{TICKER}_{TF}.csv"

Path(SIGNALS_CSV).parent.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(FEAT_TODAY).sort_values("time").reset_index(drop=True)
close_last = float(df["close"].iloc[-1])

ml_out, _   = infer_ml(ML_PKL, FEAT_TODAY, H)                 # {'P_down','P_flat','P_up','score'}
cv_out, _   = infer_cv(CV_PT, CV_MANIFEST)                     # {'P_down','P_flat','P_up','score',...}
math_out, _ = infer_math_from_prices(FEAT_TODAY, H*5, THRESH_PCT)

s_ml = float((ml_out or {}).get("score", np.nan))
s_cv = float((cv_out or {}).get("score", np.nan))
p_up = float((ml_out or {}).get("P_up", np.nan))
p_dn = float((ml_out or {}).get("P_down", np.nan))
p_up_tau = float((math_out or {}).get("p_up_tau", np.nan))

row = dict(
    time=pd.to_datetime(df["time"].iloc[-1], utc=True),
    close=close_last, s_ml=s_ml, s_cv=s_cv,
    p_up=p_up, p_down=p_dn, p_up_tau=p_up_tau
)

# аппенд в общий файл сигналов
if Path(SIGNALS_CSV).is_file():
    base = pd.read_csv(SIGNALS_CSV)
    base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
else:
    base = pd.DataFrame([row])
base.to_csv(SIGNALS_CSV, index=False)
print("appended:", row)
