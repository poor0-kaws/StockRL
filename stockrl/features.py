from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockrl.config import FeatureConfig


@dataclass(frozen=True)
class FeatureScaler:
    means: dict[str, float]
    stds: dict[str, float]


def build_features(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build indicators using only current and past prices."""
    feature_config = config or FeatureConfig()
    featured = frame.copy()

    close = featured["close"]
    high = featured["high"]
    low = featured["low"]
    volume = featured["volume"]
    previous_close = close.shift(1)

    featured["daily_return"] = close.pct_change().fillna(0.0)
    featured["volume_change"] = volume.pct_change().fillna(0.0)
    featured[f"rsi_{feature_config.rsi_window}"] = compute_rsi(close, feature_config.rsi_window)

    ema_fast = close.ewm(span=feature_config.macd_fast_window, adjust=False).mean()
    ema_slow = close.ewm(span=feature_config.macd_slow_window, adjust=False).mean()
    featured["macd"] = ema_fast - ema_slow
    featured["macd_signal"] = featured["macd"].ewm(
        span=feature_config.macd_signal_window,
        adjust=False,
    ).mean()

    featured[f"sma_{feature_config.sma_short_window}"] = close.rolling(
        window=feature_config.sma_short_window,
        min_periods=feature_config.sma_short_window,
    ).mean()
    featured[f"sma_{feature_config.sma_long_window}"] = close.rolling(
        window=feature_config.sma_long_window,
        min_periods=feature_config.sma_long_window,
    ).mean()

    featured["price_to_sma_10"] = close / featured["sma_10"]
    featured["price_to_sma_50"] = close / featured["sma_50"]
    featured["sma_gap_pct"] = (featured["sma_10"] - featured["sma_50"]) / featured["sma_50"]
    featured["volatility_20"] = featured["daily_return"].rolling(
        window=feature_config.volatility_window,
        min_periods=feature_config.volatility_window,
    ).std()
    featured["momentum_5"] = close.pct_change(periods=feature_config.momentum_window)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(
        window=feature_config.atr_window,
        min_periods=feature_config.atr_window,
    ).mean()
    featured[f"atr_{feature_config.atr_window}_pct"] = atr / close

    rolling_high = close.rolling(
        window=feature_config.regime_window,
        min_periods=feature_config.regime_window,
    ).max()
    rolling_low = close.rolling(
        window=feature_config.regime_window,
        min_periods=feature_config.regime_window,
    ).min()
    featured[f"distance_from_high_{feature_config.regime_window}"] = (close / rolling_high) - 1.0
    featured[f"distance_from_low_{feature_config.regime_window}"] = (close / rolling_low) - 1.0
    featured["regime_above_sma_50"] = (close > featured["sma_50"]).astype(float)

    featured = featured.replace([np.inf, -np.inf], np.nan)
    featured = featured.dropna().reset_index(drop=True)
    return featured


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute RSI from first principles using rolling average gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + relative_strength))

    # If average_loss is exactly zero, the series moved up every day in the window.
    # RSI should read as 100, not NaN.
    zero_loss_mask = average_loss == 0.0
    rsi = rsi.where(~zero_loss_mask, 100.0)
    return rsi


def fit_scaler(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> FeatureScaler:
    """Fit a simple z-score scaler on train data only."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for column in feature_columns:
        means[column] = float(frame[column].mean())
        std = float(frame[column].std(ddof=0))
        stds[column] = std if std > 0 else 1.0

    return FeatureScaler(means=means, stds=stds)


def apply_scaler(
    frame: pd.DataFrame,
    scaler: FeatureScaler,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Apply a previously fitted scaler without touching future data."""
    scaled = frame.copy()

    for column in feature_columns:
        scaled[column] = (scaled[column] - scaler.means[column]) / scaler.stds[column]

    return scaled


def select_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    """Return the feature matrix with a stable column order."""
    return frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
