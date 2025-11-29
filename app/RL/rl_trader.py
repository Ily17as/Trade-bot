"""Reinforcement-learning trading agent that chooses trade direction and size.

The agent consumes the enriched feature set already used across the project:
- signal (baseline discretionary signal, optional)
- tp_price / sl_price (take-profit & stop-loss levels)
- size (kelly/volatility target sizing hint)
- atr_14, logret, regime, sigma_t, arima_sign
- proba_up, proba_down, pred_class (outputs of classification model)

It learns a policy that decides whether to trade and what fraction of capital
(1–100%) to allocate. Reward is the realized PnL of the next bar when TP/SL is hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim


@dataclass
class Transition:
    state: torch.Tensor
    action_log_prob: torch.Tensor
    size_frac: torch.Tensor
    reward: torch.Tensor


FEATURE_COLUMNS = [
    "signal",
    "tp_price",
    "sl_price",
    "size",
    "atr_14",
    "logret",
    "regime",
    "sigma_t",
    "arima_sign",
    "proba_up",
    "proba_down",
    "pred_class",
]


def _safe_np(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def add_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Populate missing helper columns used by the policy.

    - ``atr_14`` is computed if absent using the standard True Range formula.
    - ``logret`` is log return of ``close``.
    - ``regime`` defaults to a 2-state proxy based on rolling volatility deciles
      when no HMM labels are provided.
    - ``sigma_t`` falls back to a rolling volatility estimate when GARCH is
      unavailable.
    - ``arima_sign`` defaults to the sign of the 1-step difference of close.
    - ``signal``/``size`` default to zeros so the agent can override them.
    - ``pred_class`` defaults to ``argmax(proba_up, 1-proba_up-proba_down, proba_down)``.
    """

    data = df.copy()

    if "atr_14" not in data.columns:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        data["atr_14"] = tr.rolling(window=14, min_periods=1).mean()

    if "logret" not in data.columns:
        data["logret"] = np.log(data["close"]).diff().fillna(0.0)

    if "regime" not in data.columns:
        vol = data["logret"].rolling(window=50, min_periods=10).std().fillna(0.0)
        low_thr, high_thr = vol.quantile([0.33, 0.66])
        data["regime"] = np.select(
            [vol <= low_thr, vol >= high_thr],
            [0, 2],
            default=1,
        )

    if "sigma_t" not in data.columns:
        data["sigma_t"] = data["logret"].rolling(window=30, min_periods=10).std().fillna(0.0)

    if "arima_sign" not in data.columns:
        data["arima_sign"] = np.sign(data["close"].diff()).fillna(0.0)

    if "signal" not in data.columns:
        data["signal"] = 0

    if "size" not in data.columns:
        data["size"] = 0.0

    if "pred_class" not in data.columns:
        mid = 1 - data.get("proba_up", 0.0) - data.get("proba_down", 0.0)
        probs = np.vstack([data.get("proba_down", 0.0), mid, data.get("proba_up", 0.0)]).T
        data["pred_class"] = np.argmax(probs, axis=1)

    for col in ("proba_up", "proba_down"):
        if col not in data.columns:
            data[col] = 0.0

    return data


class TradingEnv:
    """Single-pass environment that rewards TP/SL hits on the next bar.

    The environment iterates over each bar once (no replay buffer). The action is
    split into trade direction (long/short/flat) and allocation fraction.
    """

    def __init__(self, data: pd.DataFrame, device: Optional[torch.device] = None):
        self.data = add_technical_columns(data)
        self.device = device or torch.device("cpu")
        self.pointer = 0

    def reset(self) -> torch.Tensor:
        self.pointer = 0
        return self._get_state(self.pointer)

    def _get_state(self, idx: int) -> torch.Tensor:
        row = self.data.iloc[idx][FEATURE_COLUMNS]
        return torch.tensor(_safe_np(row.values), dtype=torch.float32, device=self.device)

    def step(self, action: int, size_frac: float) -> Tuple[torch.Tensor, float, bool]:
        entry = self.data.iloc[self.pointer]
        reward = 0.0

        # Stop at the last usable bar to keep tp/sl lookup valid.
        done = self.pointer >= len(self.data) - 1
        if done:
            return self._get_state(self.pointer), reward, True

        next_row = self.data.iloc[self.pointer + 1]
        close_next = float(next_row.get("close", entry["close"]))
        tp_price = float(entry.get("tp_price", np.nan))
        sl_price = float(entry.get("sl_price", np.nan))

        trade_dir = {0: 0, 1: 1, 2: -1}.get(action, 0)
        risk_size = max(0.0, min(1.0, float(size_frac)))

        if trade_dir != 0 and not math.isnan(tp_price) and not math.isnan(sl_price):
            # Simulate intra-bar TP/SL hit using high/low of the next bar.
            high_next = float(next_row.get("high", close_next))
            low_next = float(next_row.get("low", close_next))

            if trade_dir > 0:  # long
                tp_hit = high_next >= tp_price
                sl_hit = low_next <= sl_price
                if tp_hit and sl_hit:
                    tp_order = min(high_next, tp_price)
                    sl_order = max(low_next, sl_price)
                    # Favor first hit by price proximity.
                    tp_distance = max(tp_price - entry["close"], 1e-6)
                    sl_distance = max(entry["close"] - sl_price, 1e-6)
                    tp_priority = tp_distance / (tp_distance + sl_distance)
                    outcome = tp_order if np.random.rand() < tp_priority else sl_order
                elif tp_hit:
                    outcome = tp_price
                elif sl_hit:
                    outcome = sl_price
                else:
                    outcome = close_next
                reward = risk_size * (outcome - entry["close"]) / entry["close"]
            else:  # short
                tp_hit = low_next <= tp_price
                sl_hit = high_next >= sl_price
                if tp_hit and sl_hit:
                    tp_order = max(low_next, tp_price)
                    sl_order = min(high_next, sl_price)
                    tp_distance = max(entry["close"] - tp_price, 1e-6)
                    sl_distance = max(sl_price - entry["close"], 1e-6)
                    tp_priority = tp_distance / (tp_distance + sl_distance)
                    outcome = tp_order if np.random.rand() < tp_priority else sl_order
                elif tp_hit:
                    outcome = tp_price
                elif sl_hit:
                    outcome = sl_price
                else:
                    outcome = close_next
                reward = risk_size * (entry["close"] - outcome) / entry["close"]

        self.pointer += 1
        next_state = self._get_state(self.pointer)
        done = self.pointer >= len(self.data) - 1
        return next_state, reward, done


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, 3)  # flat / long / short
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 0..1 fraction
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.policy_head(h)
        size_frac = self.size_head(h).squeeze(-1)
        return logits, size_frac


class REINFORCEAgent:
    def __init__(self, input_dim: int, lr: float = 3e-4, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.policy = PolicyNet(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, size_frac = self.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob, size_frac

    def update(self, transitions: List[Transition], gamma: float = 0.99):
        returns = []
        g = 0.0
        for t in reversed(transitions):
            g = t.reward.item() + gamma * g
            returns.insert(0, g)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

        policy_loss = []
        size_loss = []
        for trans, ret in zip(transitions, returns_t):
            policy_loss.append(-trans.action_log_prob * ret)
            # Encourage larger sizes when returns are positive and vice versa.
            size_loss.append(-ret * torch.log(trans.size_frac + 1e-6))

        loss = torch.stack(policy_loss).sum() + 0.1 * torch.stack(size_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()


def train_reinforce(
    df: pd.DataFrame,
    episodes: int = 50,
    device: Optional[torch.device] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
) -> REINFORCEAgent:
    env = TradingEnv(df, device=device)
    agent = REINFORCEAgent(len(FEATURE_COLUMNS), device=device)

    for ep in range(episodes):
        state = env.reset()
        transitions: List[Transition] = []
        done = False
        while not done:
            action, log_prob, size_frac = agent.select_action(state)
            next_state, reward, done = env.step(action, float(size_frac.item()))
            transitions.append(
                Transition(
                    state=state,
                    action_log_prob=log_prob,
                    size_frac=size_frac.detach(),
                    reward=torch.tensor(reward, dtype=torch.float32, device=agent.device),
                )
            )
            state = next_state
        loss = agent.update(transitions)
        if progress_cb:
            progress_cb(ep, loss)
    return agent


def generate_trading_signals(agent: REINFORCEAgent, df: pd.DataFrame) -> pd.DataFrame:
    """Run a trained agent on data and append RL trade decisions."""
    env = TradingEnv(df, device=agent.device)
    actions = []
    sizes = []

    state = env.reset()
    done = False
    while not done:
        logits, size_frac = agent.policy(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = int(torch.argmax(dist.probs).item())
        actions.append(action - 1)  # map back to (-1,0,1)
        sizes.append(float(size_frac.item()))
        _, _, done = env.step(action, float(size_frac.item()))
        state = env._get_state(env.pointer)

    # The environment stops one bar early to keep TP/SL lookups valid. Pad the
    # final bar with a flat action and zero size so the output aligns with the
    # original dataframe length.
    if len(actions) < len(env.data):
        actions.append(0)
        sizes.append(0.0)

    result = env.data.copy()
    result["rl_signal"] = actions
    result["rl_size_pct"] = np.clip(np.array(sizes) * 100, 1, 100)
    return result


if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "open": [100, 101, 102, 101],
            "high": [102, 103, 103, 102],
            "low": [99, 100, 101, 99],
            "close": [101, 102, 101, 100],
            "tp_price": [102, 103, 102, 101],
            "sl_price": [99, 100, 100, 99],
            "proba_up": [0.6, 0.55, 0.4, 0.35],
            "proba_down": [0.2, 0.25, 0.35, 0.4],
        }
    )

    agent = train_reinforce(sample, episodes=3)
    enriched = generate_trading_signals(agent, sample)
    print(enriched[["close", "rl_signal", "rl_size_pct"]])