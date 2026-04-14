from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stockrl.config import DataConfig, EnvConfig, FeatureConfig, TrainConfig
from stockrl.data_loader import download_price_data, split_by_time
from stockrl.evaluate import evaluate_policy_model
from stockrl.features import apply_scaler, build_features, fit_scaler
from stockrl.trading_env import TradingEnv


def train_agent(
    data_config: DataConfig | None = None,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
    train_config: TrainConfig | None = None,
    model_out: str | Path | None = None,
):
    """Train a PPO agent on the train split and report validation metrics."""
    dataset_config = data_config or DataConfig()
    indicator_config = feature_config or FeatureConfig()
    market_config = env_config or EnvConfig()
    learning_config = train_config or TrainConfig()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for training. Install project dependencies first."
        ) from exc

    raw_prices = download_price_data(
        ticker=dataset_config.ticker,
        start=dataset_config.start,
        end=dataset_config.end,
    )
    featured = build_features(raw_prices, config=indicator_config)
    splits = split_by_time(
        featured,
        train_ratio=dataset_config.train_ratio,
        validation_ratio=dataset_config.validation_ratio,
    )

    train_frame, validation_frame = prepare_splits_for_model(
        splits.train,
        splits.validation,
        splits.test,
        indicator_config,
    )[:2]

    env = TradingEnv(train_frame, feature_config=indicator_config, env_config=market_config)
    model = PPO("MlpPolicy", env, verbose=0, seed=learning_config.random_seed)
    model.learn(total_timesteps=learning_config.total_timesteps)

    if model_out is not None:
        model_path = Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))

    validation_summary = evaluate_policy_model(
        model=model,
        frame=validation_frame,
        feature_config=indicator_config,
        env_config=market_config,
    )

    return model, validation_summary


def prepare_splits_for_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_config.scale_features:
        return train_frame, validation_frame, test_frame

    scaler = fit_scaler(train_frame, feature_config.feature_columns)
    train_scaled = apply_scaler(train_frame, scaler, feature_config.feature_columns)
    validation_scaled = apply_scaler(validation_frame, scaler, feature_config.feature_columns)
    test_scaled = apply_scaler(test_frame, scaler, feature_config.feature_columns)
    return train_scaled, validation_scaled, test_scaled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small PPO agent on SPY.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to download.")
    parser.add_argument("--start", type=str, default=None, help="Start date for downloads.")
    parser.add_argument("--end", type=str, default=None, help="End date for downloads.")
    parser.add_argument("--model-out", type=str, default="artifacts/ppo_spy.zip")
    parser.add_argument("--timesteps", type=int, default=20_000)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_config = DataConfig(ticker=args.ticker, start=args.start, end=args.end)
    train_config = TrainConfig(total_timesteps=args.timesteps)

    _, validation_summary = train_agent(
        data_config=data_config,
        train_config=train_config,
        model_out=args.model_out,
    )
    print(validation_summary.to_report_text(prefix="Validation"))


if __name__ == "__main__":
    main()
