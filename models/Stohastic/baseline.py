import numpy as np, pandas as pd

def ewma_sigma(r, lam=0.94):
    s2 = np.zeros_like(r); s2[0]=r[:100].var() if len(r)>100 else r.var()
    for t in range(1,len(r)):
        s2[t] = lam*s2[t-1] + (1-lam)*r[t-1]**2
    return np.sqrt(s2)

def gbm_mc(mu, sigma, S0, minutes, paths=20000):
    dt = 1/ (252*6.5*60)               # минута как доля года (≈ 252д * 6.5ч * 60мин)
    T  = minutes * dt
    Z  = np.random.normal(size=paths)
    ST = S0 * np.exp((mu-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    ret = ST/S0 - 1.0
    return ret

# входные данные
df = pd.read_parquet("../data/features/ml/SBER/5m/features_labeled.parquet").sort_values("time")
close = df["close"].to_numpy(dtype=float)
logret = np.diff(np.log(close))
sigma_t = ewma_sigma(logret)                    # мгновенная σ_t
mu_t = pd.Series(logret).rolling(390, min_periods=100).mean().to_numpy()  # дрейф ~день

# оценка на последнем баре и симуляция на H=180 минут
H_MIN, THR = 180, 0.02
mu_hat  = float(mu_t[-1] * (252*6.5*60))       # годовой дрейф из минутного
sigma_hat = float(sigma_t[-1] * np.sqrt(252*6.5*60))
ret = gbm_mc(mu_hat, sigma_hat, close[-1], minutes=H_MIN, paths=50000)

p_up_tau = (ret > THR).mean()
VaR95 = np.quantile(ret, 0.05)
CVaR95 = ret[ret <= VaR95].mean()
fair_score = mu_hat / (sigma_hat+1e-8)         # Sharpe-подобный скор

print(dict(p_up_tau=float(p_up_tau), VaR95=float(VaR95), CVaR95=float(CVaR95), mu=mu_hat, sigma=sigma_hat, score=fair_score))
