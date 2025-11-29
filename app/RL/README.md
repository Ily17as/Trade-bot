# RL policy for trade sizing and execution

This module adds a lightweight REINFORCE-based policy that decides **whether to trade** and **what fraction of capital (1–100%)** to allocate on each bar. It consumes the same enriched feature set already used across the project:

- `signal` — baseline discretionary signal (optional)
- `tp_price` / `sl_price` — take-profit & stop-loss for the bar
- `size` — suggested Kelly/volatility target fraction (optional)
- `atr_14`, `logret`, `regime`, `sigma_t`, `arima_sign`
- `proba_up`, `proba_down`, `pred_class` — outputs of the classification model

Missing columns are auto-filled where possible (ATR, logret, regime proxy, rolling sigma, ARIMA sign). The environment rewards the agent when its TP/SL levels are hit on the next bar; otherwise it receives the simple one-bar PnL.

## Quick start
```python
import pandas as pd
from models.RL.rl_trader import train_reinforce, generate_trading_signals

df = pd.read_csv("your_dataset.csv")
agent = train_reinforce(df, episodes=100)
results = generate_trading_signals(agent, df)
print(results[["close", "rl_signal", "rl_size_pct"]].head())
```

## Design notes
- **Action space**: 3-way direction (flat/long/short) plus a continuous size head (0–1) mapped to 1–100% of capital.
- **Reward**: outcome of hitting TP/SL on the next bar (using high/low), or fallback to next close; scaled by size.
- **Stability tricks**: feature standardization via `nan_to_num`, LayerNorm backbone, gradient clipping, and return normalization.