from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockrl.config import DataConfig, EnvConfig, FeatureConfig
from stockrl.data_loader import download_price_data, split_by_time
from stockrl.features import apply_scaler, build_features, fit_scaler
from stockrl.trading_env import TradingEnv


@dataclass(frozen=True)
class EvaluationSummary:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    final_portfolio_value: float
    benchmark_return: float
    benchmark_final_value: float

    def to_report_text(self, prefix: str = "Evaluation") -> str:
        lines = [
            f"{prefix} summary",
            f"  total_return: {self.total_return:.4f}",
            f"  annualized_return: {self.annualized_return:.4f}",
            f"  sharpe_ratio: {self.sharpe_ratio:.4f}",
            f"  max_drawdown: {self.max_drawdown:.4f}",
            f"  trade_count: {self.trade_count}",
            f"  final_portfolio_value: {self.final_portfolio_value:.2f}",
            f"  benchmark_return: {self.benchmark_return:.4f}",
            f"  benchmark_final_value: {self.benchmark_final_value:.2f}",
        ]
        return "\n".join(lines)


def evaluate_policy_model(
    model,
    frame: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
) -> EvaluationSummary:
    indicator_config = feature_config or FeatureConfig()
    market_config = env_config or EnvConfig()
    env = TradingEnv(frame=frame, feature_config=indicator_config, env_config=market_config)

    observation, _ = env.reset()
    terminated = False
    portfolio_values = [env.state.portfolio_value]

    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, _, _ = env.step(int(action))
        portfolio_values.append(env.state.portfolio_value)

    benchmark_final_value = compute_buy_and_hold_value(frame, market_config.initial_cash, market_config.fee_rate)
    benchmark_return = (benchmark_final_value / market_config.initial_cash) - 1

    return summarize_portfolio(
        portfolio_values=np.array(portfolio_values, dtype=float),
        trade_count=env.state.trade_count,
        initial_cash=market_config.initial_cash,
        benchmark_final_value=benchmark_final_value,
        benchmark_return=benchmark_return,
    )


def summarize_portfolio(
    portfolio_values: np.ndarray,
    trade_count: int,
    initial_cash: float,
    benchmark_final_value: float,
    benchmark_return: float,
) -> EvaluationSummary:
    returns = pd.Series(portfolio_values).pct_change().fillna(0.0)
    total_return = (portfolio_values[-1] / initial_cash) - 1

    periods = max(len(portfolio_values) - 1, 1)
    annualized_return = (portfolio_values[-1] / initial_cash) ** (252 / periods) - 1

    volatility = float(returns.std(ddof=0))
    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = float((returns.mean() / volatility) * np.sqrt(252))

    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    return EvaluationSummary(
        total_return=float(total_return),
        annualized_return=float(annualized_return),
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        final_portfolio_value=float(portfolio_values[-1]),
        benchmark_return=float(benchmark_return),
        benchmark_final_value=float(benchmark_final_value),
    )


def compute_buy_and_hold_value(
    frame: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
) -> float:
    if len(frame) < 2:
        raise ValueError("Need at least 2 rows to compute buy-and-hold.")

    first_execution_price = float(frame.loc[1, "open"])
    final_mark_price = float(frame.loc[len(frame) - 1, "close"])
    spendable_cash = initial_cash / (1 + fee_rate)
    shares = spendable_cash / first_execution_price
    return shares * final_mark_price


def evaluate_saved_model(
    model_path: str,
    data_config: DataConfig | None = None,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
) -> EvaluationSummary:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for evaluation. Install project dependencies first."
        ) from exc

    dataset_config = data_config or DataConfig()
    indicator_config = feature_config or FeatureConfig()
    market_config = env_config or EnvConfig()

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

    scaler = fit_scaler(splits.train, indicator_config.feature_columns)
    test_frame = splits.test
    if indicator_config.scale_features:
        test_frame = apply_scaler(test_frame, scaler, indicator_config.feature_columns)

    model = PPO.load(model_path)
    return evaluate_policy_model(
        model=model,
        frame=test_frame,
        feature_config=indicator_config,
        env_config=market_config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on SPY.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_config = DataConfig(ticker=args.ticker, start=args.start, end=args.end)
    summary = evaluate_saved_model(
        model_path=args.model_path,
        data_config=data_config,
    )
    print(summary.to_report_text(prefix="Test"))


if __name__ == "__main__":
    main()
