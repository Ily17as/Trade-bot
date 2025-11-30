from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import Dict, Any

app = FastAPI(title="Stochastic Model Service", version="1.0.0")

class PredictionRequest(BaseModel):
    data: dict  # Dictionary containing time, open, high, low, close, volume data

class PredictionResponse(BaseModel):
    p_up_tau: float
    VaR95: float
    CVaR95: float
    mu: float
    sigma: float
    score: float

def ewma_sigma(r, lam=0.94):
    """Exponentially weighted moving average for volatility"""
    s2 = np.zeros_like(r)
    s2[0] = r[:100].var() if len(r) > 100 else r.var()
    for t in range(1, len(r)):
        s2[t] = lam * s2[t-1] + (1-lam) * r[t-1]**2
    return np.sqrt(s2)

def gbm_mc(mu, sigma, S0, minutes, paths=20000):
    """Geometric Brownian Motion Monte Carlo simulation"""
    dt = 1 / (252 * 6.5 * 60)  # minute as fraction of year
    T = minutes * dt
    Z = np.random.normal(size=paths)
    ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ret = ST / S0 - 1.0
    return ret

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Run stochastic analysis using Monte Carlo simulation"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)

        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Need at least 100 data points for analysis")

        # Extract close prices
        close = df['close'].to_numpy(dtype=float)
        logret = np.diff(np.log(close))

        # Estimate parameters
        sigma_t = ewma_sigma(logret)  # instantaneous volatility
        mu_t = pd.Series(logret).rolling(390, min_periods=100).mean().to_numpy()  # drift ~1 day

        # Parameters for simulation
        H_MIN = 180  # 180 minutes ahead
        THR = 0.02   # 2% threshold for "up"

        # Get latest estimates
        mu_hat = float(mu_t[-1] * (252 * 6.5 * 60))      # annual drift from minute drift
        sigma_hat = float(sigma_t[-1] * np.sqrt(252 * 6.5 * 60))  # annual vol from minute vol

        # Run Monte Carlo simulation
        ret = gbm_mc(mu_hat, sigma_hat, close[-1], minutes=H_MIN, paths=50000)

        # Calculate metrics
        p_up_tau = float((ret > THR).mean())        # probability of >2% gain
        VaR95 = float(np.quantile(ret, 0.05))       # 5% VaR (worst 5% outcomes)
        CVaR95 = float(ret[ret <= VaR95].mean())    # Conditional VaR
        fair_score = mu_hat / (sigma_hat + 1e-8)    # Sharpe-like score

        return PredictionResponse(
            p_up_tau=p_up_tau,
            VaR95=VaR95,
            CVaR95=CVaR95,
            mu=mu_hat,
            sigma=sigma_hat,
            score=fair_score
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stochastic analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
