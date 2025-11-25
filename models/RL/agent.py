"""Wrapper around the PPO policy for deterministic inference."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .env import FEATURE_COLUMNS, PortfolioState, build_feature_frame, build_observation_window


class TradingAgent:
    """Utility class to run a trained PPO model over a price series."""

    def __init__(
        self,
        model_path: str | Path,
        data: pd.DataFrame,
        window_size: int = 32,
        initial_balance: float = 100_000.0,
    ) -> None:
        self.model = PPO.load(str(model_path))
        self.features = build_feature_frame(data)
        self.window_size = window_size
        self.initial_balance = initial_balance

    def predict_actions(self) -> pd.DataFrame:
        """Replay the trajectory with deterministic policy outputs."""

        state = PortfolioState(pointer=0, position=0, equity=self.initial_balance)
        positions = []
        equities = [self.initial_balance]

        for idx in range(len(self.features) - 1):
            obs = build_observation_window(
                self.features, state, self.window_size, self.initial_balance
            )
            action, _ = self.model.predict(np.expand_dims(obs, axis=0), deterministic=True)
            new_position = {0: 0, 1: 1, 2: -1}[int(action)]

            price_now = float(self.features.iloc[idx]["close"])
            price_next = float(self.features.iloc[idx + 1]["close"])
            price_change = (price_next - price_now) / price_now
            reward = new_position * price_change

            equity = state.equity * (1 + reward)
            positions.append(new_position)
            equities.append(equity)

            state = PortfolioState(pointer=state.pointer + 1, position=new_position, equity=equity)

        if len(positions) < len(self.features):
            positions.append(positions[-1] if positions else 0)

        result = self.features.copy()
        result = result.reset_index(drop=True)
        result = result.iloc[: len(positions)]
        result["rl_position"] = positions
        result["equity_curve"] = equities[: len(result)]
        return result

    def next_action(
        self, latest_window: pd.DataFrame, position: int = 0, equity: Optional[float] = None
    ) -> int:
        """Predict the next action for an externally supplied window of prices."""

        features = build_feature_frame(latest_window)
        state = PortfolioState(
            pointer=len(features) - 1,
            position=position,
            equity=equity if equity is not None else self.initial_balance,
        )
        obs = build_observation_window(features, state, self.window_size, self.initial_balance)
        action, _ = self.model.predict(np.expand_dims(obs, axis=0), deterministic=True)
        return int(action)


class RLAgent:
    """Thin wrapper around PPO policy with helper labels for inference."""

    ACTION_LABELS = {0: "flat", 1: "long", 2: "short"}

    def __init__(
        self,
        model_path: str | Path,
        window_size: int = 32,
        initial_balance: float = 100_000.0,
    ) -> None:
        self.model = PPO.load(str(model_path))
        self.window_size = window_size
        self.initial_balance = initial_balance

    def get_action(self, features: pd.DataFrame, state: PortfolioState) -> Tuple[int, str]:
        """Return deterministic action and a readable label for the latest window."""

        if features.empty:
            raise ValueError("RLAgent cannot act on an empty feature frame")

        observation = build_observation_window(
            features, state, self.window_size, self.initial_balance
        )
        action, _ = self.model.predict(np.expand_dims(observation, axis=0), deterministic=True)
        action_idx = int(action)
        return action_idx, self.ACTION_LABELS.get(action_idx, "unknown")


__all__ = ["TradingAgent", "RLAgent", "FEATURE_COLUMNS"]