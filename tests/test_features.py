from __future__ import annotations

import pandas as pd

from stockrl.config import FeatureConfig
from stockrl.features import apply_scaler, build_features, fit_scaler


def make_price_frame(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series(range(100, 100 + periods), dtype=float)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close + 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000,
        }
    )
    return frame


def test_build_features_creates_stable_feature_columns() -> None:
    config = FeatureConfig()
    features = build_features(make_price_frame(), config=config)

    assert list(config.feature_columns) == [
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
    ]
    assert not features.loc[:, list(config.feature_columns)].isna().any().any()


def test_scaler_uses_train_statistics_only() -> None:
    config = FeatureConfig()
    features = build_features(make_price_frame(), config=config)
    train = features.iloc[:10].reset_index(drop=True)
    validation = features.iloc[10:20].reset_index(drop=True)

    scaler = fit_scaler(train, config.feature_columns)
    scaled_validation = apply_scaler(validation, scaler, config.feature_columns)

    train_mean = train["daily_return"].mean()
    validation_mean = validation["daily_return"].mean()

    assert scaler.means["daily_return"] == train_mean
    assert scaled_validation["daily_return"].mean() != validation_mean
