from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    ticker: str = "SPY"
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    start: str | None = None
    end: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    rsi_window: int = 14
    macd_fast_window: int = 12
    macd_slow_window: int = 26
    macd_signal_window: int = 9
    sma_short_window: int = 10
    sma_long_window: int = 50
    volatility_window: int = 20
    momentum_window: int = 5
    scale_features: bool = True
    feature_columns: tuple[str, ...] = field(
        default=(
            "daily_return",
            "rsi_14",
            "macd",
            "macd_signal",
            "sma_10",
            "sma_50",
            "price_to_sma_10",
            "price_to_sma_50",
            "sma_gap_pct",
            "volatility_20",
            "momentum_5",
        )
    )


@dataclass(frozen=True)
class EnvConfig:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.001


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 20_000
    random_seed: int = 7

