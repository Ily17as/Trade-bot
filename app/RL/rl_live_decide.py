import torch, numpy as np, pandas as pd
from pathlib import Path
from inference import infer_ml, infer_cv, infer_math_from_prices
from rl_trader import QNet  # используем ту же архитектуру

# === конфиг ===
TICKER, TF, H = "SBER", "5m", 36
FEAT_TODAY = f"../data/features/ml/{TICKER}/{TF}/features_today.parquet"
CV_MANIFEST= f"../data/cv/images/{TICKER}/{TF}/today_manifest.csv"
ML_PKL     = f"../models/{TICKER}_{TF}_lgbm.pkl"
CV_PT      = f"../models/cv_resnet18_state.pt"
THRESH_PCT = 0.02
DQN_WEIGHTS= f"../models/rl_{TICKER}_{TF}_dqn.pt"  # сохранённый state_dict Q-сети
FEATURES   = ["s_ml","s_cv","s_ens","p_up","p_down","p_up_tau"]

# 1) собираем текущие признаки из inference.py
ml_out, _ = infer_ml(ML_PKL, FEAT_TODAY, H)
cv_out, _ = infer_cv(CV_PT, CV_MANIFEST)
math_out,_= infer_math_from_prices(FEAT_TODAY, H*5, THRESH_PCT)

s_ml  = float((ml_out or {}).get("score", np.nan))
s_cv  = float((cv_out or {}).get("score", np.nan))
p_up  = float((ml_out or {}).get("P_up", np.nan))
p_dn  = float((ml_out or {}).get("P_down", np.nan))
p_tau = float((math_out or {}).get("p_up_tau", np.nan))
x = pd.Series(dict(s_ml=s_ml, s_cv=s_cv, s_ens=np.nanmean([s_ml, s_cv]),
                   p_up=p_up, p_down=p_dn, p_up_tau=p_tau), index=FEATURES).astype(np.float32).to_numpy()

# 2) грузим обученную DQN-сеть и делаем действие
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q = QNet(n_obs=len(FEATURES), n_act=3).to(device)
sd = torch.load(DQN_WEIGHTS, map_location=device, weights_only=True)
q.load_state_dict(sd, strict=True)
q.eval()

with torch.no_grad():
    qv = q(torch.from_numpy(x).to(device).unsqueeze(0))   # [1,3]
    a_idx = int(torch.argmax(qv, dim=1).item())           # {0,1,2}
    action = a_idx - 1                                    # {-1,0,1}
print({"action": int(action), "side": "BUY" if action>0 else ("SELL" if action<0 else "HOLD"),
       "inputs": dict(zip(FEATURES, map(float,x)))})
