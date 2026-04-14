from __future__ import annotations

import pandas as pd
import pytest

from stockrl.config import EnvConfig, FeatureConfig
from stockrl.evaluate import (
    compute_buy_and_hold_value,
    evaluate_policy_run,
    simulate_ma_crossover_curve,
    summarize_portfolio,
)


def test_buy_and_hold_matches_manual_calculation() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [10.0, 20.0, 30.0],
            "high": [10.0, 20.0, 30.0],
            "low": [10.0, 20.0, 30.0],
            "close": [10.0, 21.0, 40.0],
            "volume": [1, 1, 1],
        }
    )

    value = compute_buy_and_hold_value(frame, initial_cash=100.0, fee_rate=0.0)
    assert value == pytest.approx(200.0)


def test_summarize_portfolio_reports_total_return_and_drawdown() -> None:
    summary = summarize_portfolio(
        portfolio_values=[100.0, 120.0, 90.0, 110.0],
        trade_count=2,
        initial_cash=100.0,
        benchmark_final_value=105.0,
        benchmark_return=0.05,
        ma_crossover_final_value=102.0,
        ma_crossover_return=0.02,
    )

    assert summary.total_return == pytest.approx(0.10)
    assert summary.max_drawdown == pytest.approx(-0.25)
    assert summary.trade_count == 2
    assert summary.ma_crossover_return == pytest.approx(0.02)


def test_ma_crossover_curve_stays_above_zero() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 12.0, 11.0, 14.0],
            "volume": [1, 1, 1, 1],
            "daily_return": [0.0, 0.2, -0.08, 0.27],
            "volume_change": [0.0, 0.0, 0.0, 0.0],
            "rsi_14": [50.0, 55.0, 45.0, 60.0],
            "macd": [0.0, 0.1, 0.1, 0.2],
            "macd_signal": [0.0, 0.05, 0.08, 0.15],
            "sma_10": [9.0, 11.0, 11.5, 13.0],
            "sma_50": [10.0, 10.5, 10.7, 11.0],
            "price_to_sma_10": [1.1, 1.1, 0.95, 1.08],
            "price_to_sma_50": [1.0, 1.14, 1.03, 1.27],
            "sma_gap_pct": [-0.1, 0.05, 0.07, 0.18],
            "volatility_20": [0.1, 0.1, 0.1, 0.1],
            "momentum_5": [0.0, 0.0, 0.0, 0.0],
            "atr_14_pct": [0.01, 0.01, 0.01, 0.01],
            "distance_from_high_20": [-0.05, 0.0, -0.08, 0.0],
            "distance_from_low_20": [0.0, 0.2, 0.1, 0.4],
            "regime_above_sma_50": [0.0, 1.0, 1.0, 1.0],
        }
    )

    curve = simulate_ma_crossover_curve(frame, initial_cash=100.0, fee_rate=0.0)
    assert len(curve) == len(frame)
    assert float(curve["portfolio_value"].iloc[-1]) > 0


def test_evaluate_policy_run_collects_trade_log() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="D"),
            "open": [10.0, 10.0, 12.0],
            "high": [10.0, 10.0, 12.0],
            "low": [10.0, 10.0, 12.0],
            "close": [10.0, 12.0, 11.0],
            "volume": [1, 1, 1],
            "daily_return": [0.0, 0.2, -0.08],
            "volume_change": [0.0, 0.0, 0.0],
            "rsi_14": [50.0, 60.0, 40.0],
            "macd": [0.0, 0.1, 0.0],
            "macd_signal": [0.0, 0.05, 0.02],
            "sma_10": [9.0, 10.0, 10.0],
            "sma_50": [8.0, 9.0, 9.5],
            "price_to_sma_10": [1.1, 1.2, 1.1],
            "price_to_sma_50": [1.25, 1.33, 1.16],
            "sma_gap_pct": [0.12, 0.11, 0.05],
            "volatility_20": [0.1, 0.1, 0.1],
            "momentum_5": [0.0, 0.0, 0.0],
            "atr_14_pct": [0.01, 0.01, 0.01],
            "distance_from_high_20": [0.0, 0.0, -0.08],
            "distance_from_low_20": [0.0, 0.2, 0.1],
            "regime_above_sma_50": [1.0, 1.0, 1.0],
        }
    )

    class BuyThenHoldModel:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, observation, deterministic=True):
            self.calls += 1
            if self.calls == 1:
                return 2, None
            return 1, None

    artifacts = evaluate_policy_run(
        model=BuyThenHoldModel(),
        frame=frame,
        feature_config=FeatureConfig(scale_features=False),
        env_config=EnvConfig(initial_cash=100.0, fee_rate=0.0),
    )

    assert len(artifacts.portfolio_curve) == len(frame)
    assert len(artifacts.trade_log) == 1
    assert artifacts.trade_log.loc[0, "action"] == "buy"
