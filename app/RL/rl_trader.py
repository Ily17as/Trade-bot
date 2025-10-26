import os, math, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ===================== CONFIG =====================
TICKER      = os.getenv("TICKER", "SBER")
TF          = os.getenv("TF", "5m")
SIGNALS_CSV = os.getenv("SIGNALS_CSV", f"../../data/signals/{TICKER}_{TF}.csv")
OUT_DECIS   = os.getenv("OUT_DECIS",   f"../../data/signals/{TICKER}_{TF}_decisions.csv")
SEED        = int(os.getenv("SEED", "42"))

# Комиссии и параметры среды
COST_BPS    = float(os.getenv("COST_BPS", "3"))     # комиссия на смену позиции, б.п.
MAX_POS     = 1                                     # {-1,0,1}
HOLD_EOD    = True                                  # закрывать позицию в конце дня
REWARD_SCALE= float(os.getenv("REWARD_SCALE", "1.0"))

# RL
EPOCHS      = int(os.getenv("EPOCHS", "8"))
STEPS_PER_E = int(os.getenv("STEPS_PER_E", "20000"))
BATCH       = int(os.getenv("BATCH", "256"))
GAMMA       = float(os.getenv("GAMMA", "0.99"))
LR          = float(os.getenv("LR", "3e-4"))
EPS_START   = float(os.getenv("EPS_START", "0.10"))
EPS_END     = float(os.getenv("EPS_END", "0.01"))
EPS_DECAY   = int(os.getenv("EPS_DECAY", "200000"))
TARGET_TAU  = float(os.getenv("TARGET_TAU", "0.005"))   # soft update τ

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ===================== DATA =====================
def load_signals(path_csv: str) -> pd.DataFrame:
    if not Path(path_csv).is_file():
        raise FileNotFoundError(f"signals not found: {path_csv}")
    df = pd.read_csv(path_csv)
    # ожидаемые колонки:
    # time, close, s_ml, s_cv, p_up, p_down, p_up_tau (math), day (опц.)
    need = {"time","close","s_ml","s_cv"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"missing columns in {path_csv}: {missing}")

    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    # derive
    df["ret1"] = np.log(df["close"]).diff().fillna(0.0)
    if "p_up" not in df:   df["p_up"] = np.clip((df["s_ml"]+df["s_cv"])*0 + 0.5, 0, 1)
    if "p_down" not in df: df["p_down"] = 1.0 - df["p_up"]
    if "p_up_tau" not in df: df["p_up_tau"] = 0.5
    df["s_ens"] = 0.5*df["s_ml"] + 0.5*df["s_cv"]
    if "day" not in df:
        df["day"] = df["time"].dt.tz_convert("Europe/Moscow").dt.date.astype(str)
    return df

DF = load_signals(SIGNALS_CSV)

# train/val split: последний торговый день = вал/OOS
last_day = DF["day"].iloc[-1]
DF_TR = DF[DF["day"] < last_day].reset_index(drop=True)
DF_VA = DF[DF["day"] == last_day].reset_index(drop=True)
print(f"[data] train rows={len(DF_TR)} days={DF_TR['day'].nunique()}  val rows={len(DF_VA)} day={last_day}")

FEATURES = ["s_ml","s_cv","s_ens","p_up","p_down","p_up_tau"]
XTR = DF_TR[FEATURES].to_numpy(np.float32)
RTR = DF_TR["ret1"].to_numpy(np.float32)
DTR = DF_TR["day"].to_numpy()
XVA = DF_VA[FEATURES].to_numpy(np.float32)
RVA = DF_VA["ret1"].to_numpy(np.float32)
DVA = DF_VA["day"].to_numpy()

# ===================== ENV =====================
class TradingEnv:
    """Дискретные действия {-1,0,1}. Состояние = текущие сигналы (FEAT).
       Reward_t = pos_{t-1} * ret_t - cost*|pos_t - pos_{t-1}|."""
    def __init__(self, X, ret, day, cost_bps=3.0, hold_eod=True):
        self.X = X; self.ret = ret; self.day = day
        self.n = len(ret)
        self.cost = cost_bps*1e-4
        self.hold_eod = hold_eod
        self.reset()

    def reset(self, idx=None):
        self.t = 1 if idx is None else max(1, idx)  # чтобы была ret_t
        self.pos = 0
        return self._obs()

    def _obs(self):
        return self.X[self.t].copy()

    def step(self, action: int):
        action = int(np.clip(action, -MAX_POS, MAX_POS))
        # комиссия за смену позиции
        switch_cost = self.cost * abs(action - self.pos)
        # дневной конец — принудительный флаттен
        if self.hold_eod and self.day[self.t] != self.day[self.t-1]:
            # закрываем позицию со стоимостью
            switch_cost += self.cost * abs(0 - self.pos)
            self.pos = 0
        reward = self.pos * float(self.ret[self.t]) - switch_cost
        self.pos = action
        self.t += 1
        done = (self.t >= self.n-1)
        return self._obs(), REWARD_SCALE*reward, done

# ===================== DQN =====================
N_OBS = len(FEATURES)
N_ACT = 3  # {-1,0,1}

def to_action(idx):   # 0->-1, 1->0, 2->1
    return idx - 1

def from_action(a):   # -1->0, 0->1, 1->2
    return a + 1

class QNet(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_act)
        )
    def forward(self, x): return self.net(x)

class Replay:
    def __init__(self, cap=200_000):
        self.s, self.a, self.r, self.ns, self.d = [], [], [], [], []
        self.cap=cap; self.ptr=0
    def push(self, s,a,r,ns,d):
        if len(self.s) < self.cap:
            self.s.append(None); self.a.append(None); self.r.append(None); self.ns.append(None); self.d.append(None)
        self.s[self.ptr]=s; self.a[self.ptr]=a; self.r[self.ptr]=r; self.ns[self.ptr]=ns; self.d[self.ptr]=d
        self.ptr = (self.ptr+1) % self.cap
    def sample(self, n):
        idx = np.random.randint(0, len(self), size=n)
        s  = torch.from_numpy(np.stack([self.s[i] for i in idx])).float()
        a  = torch.tensor([self.a[i] for i in idx], dtype=torch.long)
        r  = torch.tensor([self.r[i] for i in idx], dtype=torch.float32)
        ns = torch.from_numpy(np.stack([self.ns[i] for i in idx])).float()
        d  = torch.tensor([self.d[i] for i in idx], dtype=torch.float32)
        return s,a,r,ns,d
    def __len__(self): return len(self.s)

def soft_update(target, online, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

# ===================== TRAIN =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q = QNet(N_OBS, N_ACT).to(device)
qt= QNet(N_OBS, N_ACT).to(device)
qt.load_state_dict(q.state_dict())
opt = torch.optim.AdamW(q.parameters(), lr=LR)

env = TradingEnv(XTR, RTR, DTR, cost_bps=COST_BPS, hold_eod=HOLD_EOD)
rb  = Replay()

global_step=0
eps = EPS_START

def policy(obs):
    global eps, global_step
    if random.random() < eps:
        return random.randrange(N_ACT)
    with torch.no_grad():
        qv = q(torch.from_numpy(obs).float().to(device).unsqueeze(0))
        return int(torch.argmax(qv, dim=1).item())
def anneal():
    global eps, global_step
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-global_step / EPS_DECAY)

# bootstrap replay
obs = env.reset()
for _ in range(2048):
    a = random.randrange(N_ACT)
    nobs, r, d = env.step(to_action(a))
    rb.push(obs, a, r, nobs, float(d))
    obs = nobs if not d else env.reset()

loss_hist=[]
for ep in range(EPOCHS):
    obs = env.reset()
    ep_ret = 0.0
    for _ in range(STEPS_PER_E):
        a = policy(obs)
        nobs, r, done = env.step(to_action(a))
        rb.push(obs, a, r, nobs, float(done))
        obs = nobs
        ep_ret += r
        global_step += 1; anneal()

        if len(rb) >= 2048:
            s,a,r,ns,d = rb.sample(BATCH)
            s,ns = s.to(device), ns.to(device)
            a,r,d = a.to(device), r.to(device), d.to(device)

            q_sa  = q(s).gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                a_star = torch.argmax(q(ns), dim=1, keepdim=True)
                q_tgt  = qt(ns).gather(1, a_star).squeeze(1)
                y = r + GAMMA * (1.0 - d) * q_tgt
            loss = nn.SmoothL1Loss()(q_sa, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 1.0); opt.step()
            soft_update(qt, q, TARGET_TAU)

        if done:
            obs = env.reset()
    loss_hist.append(float(loss.detach().cpu().item()) if len(rb)>=2048 else float("nan"))
    print(f"[train] epoch={ep} eps={eps:.3f} last_loss={loss_hist[-1]:.5f} ep_ret={ep_ret:.4f}")

# ===================== EVAL (последний день) =====================
env_va = TradingEnv(XVA, RVA, DVA, cost_bps=COST_BPS, hold_eod=HOLD_EOD)
obs = env_va.reset()
pos_hist, pnl_hist, time_hist, act_hist = [], [], [], []
cum = 0.0; pos = 0
for t in range(len(XVA)-1):
    with torch.no_grad():
        qv = q(torch.from_numpy(obs).float().to(device).unsqueeze(0))
        a = int(torch.argmax(qv, dim=1).item())
    act = to_action(a)
    nobs, r, done = env_va.step(act)
    pos = act
    cum += r
    # лог
    pos_hist.append(pos); pnl_hist.append(cum); time_hist.append(DF_VA["time"].iloc[env_va.t])
    act_hist.append(act)
    obs = nobs
    if done: break

eva = pd.DataFrame({"time": time_hist, "pos": pos_hist, "equity": pnl_hist, "act": act_hist})
eva.to_csv(OUT_DECIS, index=False)
print(f"[eval] saved decisions -> {OUT_DECIS}")
print(f"[eval] final equity={cum:.6f}, trades={int(np.sum(np.abs(np.diff([0]+act_hist))))}")

# ===================== SIMPLE SIGNAL WRAPPER FOR TODAY =====================
# Последнее решение как «онлайн»-сигнал
if len(eva):
    last = eva.iloc[-1]
    side = "BUY" if last["act"]>0 else ("SELL" if last["act"]<0 else "HOLD")
    print(json.dumps({"last_time": str(last["time"]), "last_action": int(last["act"]), "side": side,
                      "last_pos": int(last["pos"]), "equity": float(last["equity"])}, ensure_ascii=False))

DQN_OUT = os.getenv("DQN_OUT", f"../../models/rl_{TICKER}_{TF}_dqn.pt")
Path(DQN_OUT).parent.mkdir(parents=True, exist_ok=True)
torch.save(q.state_dict(), DQN_OUT)
print(f"[save] DQN weights -> {DQN_OUT}")
