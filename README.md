# Trade-Bot

An automated stock market trading bot built on ML/DL models.

> The goal is to assess stocks (undervalued / fairly valued /
> overvalued) and support decision‑making or enable automated trading.

------------------------------------------------------------------------

## Key Ideas

The project combines several analytical approaches:

-   **Deep Learning** for time‑series price and volume analysis\
-   **Computer Vision** for analyzing price charts as images\
-   **Statistical / stochastic models** such as Monte Carlo simulations\
-   **Reinforcement Learning** for optimizing trading strategies

The core value is precise fair‑value estimation and higher confidence in
trading decisions.

------------------------------------------------------------------------

## Project Structure

    Trade-bot/
      app/          # Application code, helper scripts, service logic
      models/       # Notebooks and/or saved model weights (DL, CV, RL, etc.)
      dataset/      # Raw and processed datasets
      requirements.txt
      LICENSE       # MIT

Most prototypes and experiments are implemented in Jupyter notebooks.

------------------------------------------------------------------------

## Features (planned / partially implemented)

-   Historical market data processing\
-   Feature generation and chart creation for multiple model types\
-   Model training:
    -   DL models on time series\
    -   CV models on chart images\
    -   Stochastic market simulation models\
    -   RL agent for trading\
-   Evaluation metrics:
    -   Precision / Recall\
    -   CAGR\
    -   Sharpe Ratio\
    -   Max Drawdown\
    -   Equity curves\
-   Model integration into a unified pipeline, future API trading
    support

------------------------------------------------------------------------

## Installation

Python 3.10+ required.

``` bash
git clone https://github.com/Ily17as/Trade-bot.git
cd Trade-bot

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

------------------------------------------------------------------------

## Working With Data

1.  Load or prepare historical stock data and place it in `dataset/`.\
2.  Prepare data:
    -   cleaning and normalization\
    -   feature engineering\
    -   dataset formatting per model type (DL, CV, RL)\
3.  Use notebooks in `app/` and `models/` for:
    -   visualization\
    -   exploratory analysis\
    -   dataset creation

Details are available inside individual notebooks.

CV dataset: https://www.kaggle.com/datasets/galievilyas/sber0601-1001

------------------------------------------------------------------------

## Training Models

General workflow:

1.  Start Jupyter:

    ``` bash
    jupyter notebook
    ```

2.  Open the required notebook.\

3.  Set dataset paths.\

4.  Run cells sequentially:

    -   load & preprocess data\
    -   model definition & training\
    -   saving weights & metrics

Retraining requires updated data in `dataset/`.

------------------------------------------------------------------------

## Running the Bot

The project is currently focused on model development.\
Full trading bot logic and broker API integration are under development.

Planned workflow:

1.  Load saved models (DL, CV, stochastic, RL).\
2.  Fetch market data in near real‑time.\
3.  Perform:
    -   stock valuation\
    -   signal generation\
    -   automated order placement (future)

Updates will be added as development progresses.

------------------------------------------------------------------------

## Roadmap

-   [X] Standardize dataset formats\
-   [X] Core DL pipeline for time series\
-   [X] CV model for chart analysis\
-   [X] Statistical models and Monte Carlo\
-   [X] RL agent\
-   [ ] Backtesting engine\
-   [ ] Broker API integration\
-   [ ] Production‑mode trading & monitoring

------------------------------------------------------------------------

## Team

-   **Daria Alexandrova** --- Computer Vision\
-   **Kamilya Shakirova** --- Deep Learning\
-   **Ilyas Galiev** --- Statistical & stochastic models, architecture

------------------------------------------------------------------------

## Contributing

1.  Fork and create a feature branch.\
2.  Follow directory structure.\
3.  Include a short description and example (notebook or script).

------------------------------------------------------------------------

## License

MIT License.
