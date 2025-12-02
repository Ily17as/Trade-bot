# Trade-bot

Experimental trading-bot framework built around **machine learning**, **computer vision**, **reinforcement learning**, and **stochastic modelling** to help evaluate stocks  
(undervalued / fairly valued / overvalued) and eventually support **automated trading decisions**.

> ⚠️ Educational / research project. Not investment advice. Use at your own risk and start with paper trading.

---

## 1. Overview

The goal of this repository is to build an end-to-end research pipeline for:

- ingesting and cleaning **stock market data**
- computing **technical indicators** and other features
- training **ML/DL models** (tabular + time series)
- training **CV models** on **candlestick chart images**
- training **RL-agent** for trading decisions
- running **stochastic simulations** and risk analysis
- plugging everything into a **backtest + (future) live trading** loop

Architecturally it is closer to an **experiment playground** than a production system: most logic is in Jupyter notebooks and Python scripts, with Airflow + Docker and basic services gradually turning it into a more structured framework.

---

## 2. Repository structure

```text
Trade-bot/
├── airflow/          # Airflow configuration and DAGs for scheduled jobs (data, training, backtests)
├── app/              # Application-level code, utilities, notebooks and prototyping scripts
├── dataset/          # Raw and processed data, helpers for building datasets
├── frontend/         # Frontend / dashboard code (visualization of signals, equity curves, metrics)
├── models/           # Trained models + training notebooks (ML, CV, RL, stochastic)
├── services/         # Python services: backtesting, live / paper-trading, data sync, etc.
├── docker-compose.yml
├── requirements.txt
└── LICENSE
```

High-level idea:

- **`dataset/`** – everything about data: loading, caching, feature generation.
- **`models/`** – model training and saved weights.
- **`services/`** – how models are actually used (backtests, inference loops).
- **`airflow/` + `docker-compose.yml`** – orchestration and infra for running things on a schedule.
- **`frontend/`** – human-friendly view of what the bot is doing.

---

## 3. Core components

### 3.1 Baseline indicators and features

The baseline feature set combines standard **technical indicators** and simple statistics, for example:

- Moving Averages (SMA/EMA) with several windows
- Volatility measures (rolling standard deviation, ATR-like features)
- Momentum / rate-of-change features
- Volume-related signals
- Candle-level features (open–high–low–close relationships, patterns)

These indicators are used in two ways:

1. As **direct decision rules / sanity checks** in the baseline strategies.
2. As **features for ML / RL models**, giving them a richer state representation.

### 3.2 ML / DL models (tabular & time series)

Classical ML & deep learning models are used to predict:

- short-term returns or probability of upward / downward move
- “fair value” zones (undervalued / fairly valued / overvalued)

Typical tasks:

- **Classification:** buy / hold / sell or over/under-valued classes.
- **Regression:** expected return, target price, risk-adjusted score.

These models are trained on handcrafted features, indicators, and optionally encoded time windows of OHLCV.

### 3.3 Computer Vision models (candlestick images)

The CV part treats **candlestick charts as images**:

- generate chart images from historical candles (e.g., fixed-length windows)
- train a CNN (with optional SE/attention blocks) to classify patterns:
  - bullish / bearish regimes
  - “good opportunity” vs “noise”
  - specific local patterns the CNN learns automatically

This gives a complementary signal: **“does the chart *look* like a profitable setup?”**, independent of raw numeric features.

### 3.4 Stochastic & statistical models

The stochastic component focuses on:

- **Monte Carlo simulations** of future price paths based on historical returns
- estimation of **risk metrics**:
  - drawdowns
  - distribution of final equity
  - probability of hitting certain thresholds
- potential **regime models** (volatility regimes, mean-reversion vs trend, etc.)

These models are not placing trades directly; instead they:

- give **scenario-based risk estimates**,
- help calibrate **position sizing**, **take-profit/stop-loss** distances,
- provide additional features / priors for RL and ML components.

### 3.5 Reinforcement Learning (RL)

The RL component (e.g. PPO-based agent) operates on a state that may include:

- recent OHLCV window
- technical indicators
- CV model scores
- stochastic / risk features (e.g., realized volatility, drawdown information)

**Action space** (examples):

- open / close position
- adjust position size (fraction of capital)
- optionally choose between “no trade”, “long”, “flat”, “reduce size”, etc.

**Reward design**:

- primarily based on **equity curve** / PnL
- augmented with **risk penalties**:
  - drawdown
  - excessive turnover
  - large position in high-volatility regimes

Currently RL is used as **an experimental layer**, not as a fully autonomous live trader. It is mainly trained and evaluated in controlled backtest environments.

---

## 4. Installation

### 4.1 Local (Python)

Requirements:

- Python 3.10+
- `pip` and (optionally) `virtualenv` / `venv`

```bash
git clone https://github.com/Ily17as/Trade-bot.git
cd Trade-bot

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scriptsctivate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Docker / Docker Compose

A `docker-compose.yml` is provided to simplify running the stack (backend services, Airflow, etc.).

Typical flow (from repository root):

```bash
# Build and start all services defined in docker-compose.yml
docker compose up --build
```

Check the compose file for defined services (Airflow, backend, DB, etc.) and adjust ports / volumes according to your environment.

---

## 5. Configuration

Create a `.env` file (or set environment variables) for sensitive configuration:

Typical values:

- **Broker / data provider:**
  - API token
  - base URL
  - instrument symbol (e.g. `SBER`, `AAPL`, etc.)
- **Data settings:**
  - timeframe (e.g. 5m, 1h)
  - history length for training
- **Storage:**
  - paths for datasets and models
- **Experiment flags:**
  - enable/disable CV model
  - enable/disable RL agent
  - risk limits (max position size, max daily loss, etc.)

Exact variable names are defined in the corresponding service and notebook code.

---

## 6. Typical workflow

This is an example end-to-end workflow using this repository.

### 6.1 Prepare data

1. Download historical data from your broker / provider  
   (e.g., via a dedicated loader in `services/` or `app/`).
2. Store cleaned data under `dataset/` (CSV/Parquet/…).
3. Optionally build:
   - time-series datasets,
   - chart images for CV models,
   - feature tables for ML models.

Some notebooks in `app/` and `models/` demonstrate data preprocessing and feature engineering.

### 6.2 Train ML / CV models

- Open Jupyter:

  ```bash
  jupyter notebook
  ```

- Go to the relevant notebook under `models/`:
  - ML / DL for time series
  - CV model for chart images
- Configure:
  - input paths under `dataset/`
  - training hyperparameters
- Run the notebook to:
  - train the model,
  - log metrics,
  - save weights into `models/`.

### 6.3 Train / evaluate RL agent

- Use RL notebooks / scripts in `models/`:
  - define environment (state, actions, reward),
  - plug in indicators and other signals,
  - train PPO (or another algorithm),
  - store the trained policy.

Evaluation:

- Run episodic backtests in the RL environment.
- Log:
  - equity curves,
  - distribution of returns,
  - risk metrics (max drawdown, Sharpe-like ratios).

### 6.4 Run backtest with the full stack

Scripts/notebooks in `services/` connect the pieces:

- Load ML, CV and RL models.
- Step through historical data candle-by-candle.
- At each step:
  - compute indicators and features,
  - get ML forecast / CV score,
  - get RL action (if enabled),
  - apply risk overlays (position sizing, hard stops),
  - simulate order execution and update portfolio state.

Outputs:

- equity curve over time,
- trade log (entry/exit, PnL per trade),
- summary statistics (CAGR, Sharpe, max drawdown, hit rate, etc.).

### 6.5 Run (future) live / paper trading

Live / paper trading services (also under `services/`) are planned / evolving to:

- periodically fetch latest candles from broker API,
- update state & features,
- query trained models and RL policy,
- decide whether to send orders,
- record all actions for later analysis.

For now, this is **experimental** and should only be used in paper-trading mode with very small or virtual capital.

---

## 7. Airflow integration

The `airflow/` directory and Docker setup are intended to orchestrate:

- periodic data ingestion jobs,
- scheduled model retraining,
- nightly backtests,
- report generation (e.g., export metrics or charts to the frontend / storage).

Typical usage pattern (conceptually):

1. Start the Airflow stack via Docker Compose.
2. Open the Airflow UI.
3. Enable DAGs for:
   - data updates,
   - retraining pipelines,
   - evaluation / backtesting.

The exact DAG names and parameters are defined inside `airflow/`.

---

## 8. Frontend

The `frontend/` folder contains code for a simple web UI to:

- visualize price series and indicators,
- display equity curves and backtest statistics,
- inspect trades (timeline / trade log table),
- show model and RL metrics.

This is not required for running the core logic but makes it easier to **interpret results** and present them.

---

## 9. Current status & roadmap

**Current state:**

- Repository is **research-oriented**:
  - Many components live in notebooks.
  - Several models are experimental.
  - Live trading is not production-ready.
- Core pieces exist for:
  - data ingestion and preprocessing,
  - feature and indicator computation,
  - ML & CV model training,
  - RL environment and initial agents,
  - backtesting logic.

**Planned / ongoing work:**

- unify dataset interfaces across ML, CV and RL parts
- more robust backtesting engine (fees, slippage, multiple instruments)
- better risk overlays (volatility-based position sizing, regime detection)
- richer RL experiments:
  - multi-asset,
  - different reward structures,
  - constrained RL with risk targets
- more polished Airflow DAGs for fully scheduled pipelines
- more polished frontend for monitoring and debugging

---

## 10. Contributing

Contributions, issue reports and ideas are welcome:

1. Fork the repository.
2. Create a feature branch.
3. Follow the existing directory structure (dataset → models → services).
4. Add tests or minimal examples where possible.
5. Open a pull request and describe:
   - what you changed,
   - how to run/check it.

---

## 11. License

This project is licensed under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.
