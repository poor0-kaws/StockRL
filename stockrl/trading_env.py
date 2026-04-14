from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stockrl.config import EnvConfig, FeatureConfig
from stockrl.features import select_feature_matrix
from stockrl.portfolio_core import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    POSITION_LONG,
    make_initial_state,
    apply_action,
)


class TradingEnv(gym.Env[np.ndarray, int]):
    """
    Small long-only trading environment.

    Timing model:
      observe day t
      act on day t + 1 open
      mark portfolio on day t + 1 close
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_config: FeatureConfig | None = None,
        env_config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.feature_config = feature_config or FeatureConfig()
        self.env_config = env_config or EnvConfig()
        self.frame = frame.reset_index(drop=True).copy()

        if len(self.frame) < 2:
            raise ValueError("TradingEnv needs at least 2 rows.")

        self.feature_matrix = select_feature_matrix(self.frame, self.feature_config.feature_columns)
        self.action_space = spaces.Discrete(3)

        observation_size = len(self.feature_config.feature_columns) + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.current_index = 0
        self.state = make_initial_state(self.env_config.initial_cash)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_index = 0
        self.state = make_initial_state(self.env_config.initial_cash)
        observation = self._build_observation(self.current_index)
        info = self._build_info()
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown action: {action}")

        previous_position = self.state.position
        next_index = self.current_index + 1
        execution_price = float(self.frame.loc[next_index, "open"])
        mark_price = float(self.frame.loc[next_index, "close"])
        previous_value = self.state.portfolio_value

        self.state = apply_action(
            state=self.state,
            action=action,
            execution_price=execution_price,
            mark_price=mark_price,
            fee_rate=self.env_config.fee_rate,
        )

        reward = float(self.state.portfolio_value - previous_value)
        self.current_index = next_index
        terminated = self.current_index >= len(self.frame) - 1
        truncated = False

        if terminated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._build_observation(self.current_index)

        info = self._build_info()
        info["last_action"] = int(action)
        info["reward"] = reward
        info["execution_price"] = execution_price
        info["mark_price"] = mark_price
        info["previous_position"] = int(previous_position)
        info["executed_trade"] = previous_position != self.state.position
        return observation, reward, terminated, truncated, info

    def _build_observation(self, index: int) -> np.ndarray:
        features = self.feature_matrix[index]
        position = np.array([float(self.state.position)], dtype=np.float32)
        return np.concatenate([features, position], dtype=np.float32)

    def _build_info(self) -> dict[str, Any]:
        return {
            "index": self.current_index,
            "date": self.frame.loc[self.current_index, "date"],
            "portfolio_value": float(self.state.portfolio_value),
            "cash": float(self.state.cash),
            "shares": float(self.state.shares),
            "position": int(self.state.position),
            "trade_count": int(self.state.trade_count),
        }


def action_name(action: int) -> str:
    if action == ACTION_SELL:
        return "sell"

    if action == ACTION_HOLD:
        return "hold"

    if action == ACTION_BUY:
        return "buy"

    raise ValueError(f"Unknown action: {action}")


def position_name(position: int) -> str:
    if position == POSITION_LONG:
        return "long"

    return "flat"
