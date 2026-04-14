from __future__ import annotations

import pandas as pd
import pytest

from stockrl.config import EnvConfig, FeatureConfig
from stockrl.trading_env import TradingEnv


def make_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.0, 12.0, 11.0],
            "volume": [100, 100, 100],
            "daily_return": [0.0, 0.01, -0.01],
            "volume_change": [0.0, 0.0, 0.0],
            "rsi_14": [50.0, 55.0, 45.0],
            "macd": [0.1, 0.2, 0.1],
            "macd_signal": [0.1, 0.15, 0.12],
            "sma_10": [10.0, 10.5, 10.8],
            "sma_50": [9.8, 10.0, 10.1],
            "price_to_sma_10": [1.0, 1.1, 1.0],
            "price_to_sma_50": [1.02, 1.2, 1.08],
            "sma_gap_pct": [0.02, 0.05, 0.07],
            "volatility_20": [0.1, 0.1, 0.1],
            "momentum_5": [0.0, 0.03, 0.01],
            "atr_14_pct": [0.01, 0.01, 0.01],
            "distance_from_high_20": [-0.02, 0.0, -0.04],
            "distance_from_low_20": [0.02, 0.05, 0.03],
            "regime_above_sma_50": [1.0, 1.0, 1.0],
        }
    )


def test_env_uses_next_bar_execution_and_reward_matches_portfolio_delta() -> None:
    env = TradingEnv(
        frame=make_feature_frame(),
        feature_config=FeatureConfig(scale_features=False),
        env_config=EnvConfig(initial_cash=100.0, fee_rate=0.0),
    )

    _, info = env.reset()
    assert info["index"] == 0

    _, reward, terminated, _, info = env.step(2)

    assert terminated is False
    assert info["index"] == 1
    assert info["position"] == 1
    expected_return = (100.0 * (12.0 / 11.0) / 100.0) - 1.0
    expected_reward = expected_return - env.env_config.trade_penalty
    assert reward == pytest.approx(expected_reward)


def test_env_terminates_on_last_tradable_step() -> None:
    env = TradingEnv(
        frame=make_feature_frame(),
        feature_config=FeatureConfig(scale_features=False),
        env_config=EnvConfig(initial_cash=100.0, fee_rate=0.0),
    )

    env.reset()
    env.step(1)
    _, _, terminated, _, _ = env.step(1)
    assert terminated is True
