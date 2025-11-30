"""Custom Gymnasium environment for PPO-based trading tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


FEATURE_COLUMNS = [
    "close",
    "log_return",
    "sma_ratio",
    "rsi",
    "atr",
    "volatility",
    "volume_zscore",
]


@dataclass
class PortfolioState:
    pointer: int
    position: int
    equity: float


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature dataframe ready for the trading environment."""

    data = df.copy()
    for col in ("open", "high", "low", "close"):
        if col not in data.columns:
            raise ValueError(f"Missing required price column: {col}")

    data.sort_index(inplace=True)
    data = data.reset_index(drop=True)

    if "volume" not in data.columns:
        data["volume"] = 0.0

    data["log_return"] = np.log(data["close"]).diff().fillna(0.0)
    data["volatility"] = (
        data["log_return"].rolling(window=20, min_periods=5).std().fillna(0.0)
    )

    sma_fast = data["close"].rolling(window=5, min_periods=1).mean()
    sma_slow = data["close"].rolling(window=20, min_periods=1).mean()
    data["sma_ratio"] = (sma_fast / sma_slow - 1).fillna(0.0)

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    data["rsi"] = 100 - (100 / (1 + rs))

    tr = _true_range(data["high"], data["low"], data["close"])
    data["atr"] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)

    volume_mean = data["volume"].rolling(window=30, min_periods=1).mean()
    volume_std = data["volume"].rolling(window=30, min_periods=1).std().replace(0, 1)
    data["volume_zscore"] = ((data["volume"] - volume_mean) / volume_std).fillna(0.0)

    return data


def build_observation_window(
    features: pd.DataFrame,
    state: PortfolioState,
    window_size: int,
    initial_equity: float,
) -> np.ndarray:
    """Create a windowed observation aligned with the environment's logic."""

    start = max(0, state.pointer - window_size + 1)
    window = features.iloc[start : state.pointer + 1]

    # После reindex появятся строки, которых нет в features → NaN.
    padded = window.reindex(range(state.pointer - window_size + 1, state.pointer + 1))

    # КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: заполняем NaN нулями, потом в numpy.
    obs_window = (
        padded[FEATURE_COLUMNS]
        .fillna(0.0)                      # <-- вот этого не хватало
        .to_numpy(dtype=np.float32)
    )

    if len(obs_window) < window_size:
        pad_rows = window_size - len(obs_window)
        obs_window = np.concatenate(
            [np.zeros((pad_rows, obs_window.shape[1]), dtype=np.float32), obs_window],
            axis=0,
        )

    position_col = np.full((window_size, 1), state.position, dtype=np.float32)
    equity_ratio = state.equity / initial_equity if initial_equity else 0.0
    equity_col = np.full((window_size, 1), equity_ratio, dtype=np.float32)

    return np.concatenate([obs_window, position_col, equity_col], axis=1)


class TradingEnv(Env):
    """Simple trading environment with windowed observations and equity tracking."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 32,
        initial_balance: float = 100_000.0,
        trading_fee: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee

        self.features = build_feature_frame(data)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_shape = (self.window_size, len(FEATURE_COLUMNS) + 2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        self.state: PortfolioState
        self._latest_observation: Optional[np.ndarray] = None
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = PortfolioState(pointer=0, position=0, equity=self.initial_balance)
        self._latest_observation = build_observation_window(
            self.features, self.state, self.window_size, self.initial_balance
        )
        return self._latest_observation, {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Приводим к скаляру и режем в [-1, 1]
        action_value = float(np.array(action).squeeze())
        action_value = float(np.clip(action_value, -1.0, 1.0))

        current_idx = self.state.pointer
        done = current_idx >= len(self.features) - 1
        if done:
            return self._latest_observation, 0.0, True, False, {"equity": self.state.equity}

        next_idx = current_idx + 1
        price_now = float(self.features.iloc[current_idx]["close"])
        price_next = float(self.features.iloc[next_idx]["close"])

        # -1..1 = доля капитала в позиции (от полной шорта до полной лонга)
        new_position = action_value

        # Можно отрезать совсем маленькие позиции как flat:
        if abs(new_position) < 1e-3:
            new_position = 0.0

        price_change = (price_next - price_now) / price_now
        reward = new_position * price_change  # доля * доходность

        # Комиссия, если сильно изменили позицию (опционально)
        if self.trading_fee > 0 and np.sign(new_position) != np.sign(self.state.position):
            reward -= self.trading_fee

        equity = self.state.equity * (1 + reward)
        self.state = PortfolioState(pointer=next_idx, position=new_position, equity=equity)

        observation = build_observation_window(
            self.features, self.state, self.window_size, self.initial_balance
        )
        self._latest_observation = observation

        terminated = self.state.pointer >= len(self.features) - 1
        return observation, float(reward), terminated, False, {"equity": equity}

    def render(self):  # pragma: no cover - placeholder for potential future UI
        return None