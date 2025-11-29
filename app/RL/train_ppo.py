"""Script to train a PPO trading agent using the custom environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import TradingEnv, build_feature_frame


def load_price_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    expected_cols = {"open", "high", "low", "close"}
    missing = expected_cols - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return data


def make_env(data: pd.DataFrame, window_size: int, initial_balance: float, trading_fee: float):
    def _init():
        return TradingEnv(
            data=data,
            window_size=window_size,
            initial_balance=initial_balance,
            trading_fee=trading_fee,
        )

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO trading agent")
    parser.add_argument("--data", required=True, help="Path to CSV file with OHLCV data")
    parser.add_argument("--window-size", type=int, default=32, help="Sliding window length")
    parser.add_argument(
        "--initial-balance", type=float, default=100_000.0, help="Starting equity for the agent"
    )
    parser.add_argument("--trading-fee", type=float, default=0.0, help="Per-trade fee")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of PPO training timesteps",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/rl/checkpoints/ppo_trading_agent.zip"),
        help="Where to save the trained agent",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_data = load_price_data(args.data)
    feature_df = build_feature_frame(raw_data)

    env = DummyVecEnv(
        [
            make_env(
                feature_df,
                window_size=args.window_size,
                initial_balance=args.initial_balance,
                trading_fee=args.trading_fee,
            )
        ]
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.checkpoint)
    print(f"Model saved to {args.checkpoint}")


if __name__ == "__main__":
    main()