from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stockrl.config import DataConfig, EnvConfig, FeatureConfig
from stockrl.data_loader import download_price_data, split_by_time
from stockrl.features import apply_scaler, build_features, fit_scaler
from stockrl.portfolio_core import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    POSITION_LONG,
    apply_action,
    make_initial_state,
)
from stockrl.trading_env import TradingEnv, action_name, position_name


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
    ma_crossover_return: float
    ma_crossover_final_value: float

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
            f"  ma_crossover_return: {self.ma_crossover_return:.4f}",
            f"  ma_crossover_final_value: {self.ma_crossover_final_value:.2f}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class EvaluationArtifacts:
    summary: EvaluationSummary
    portfolio_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    ma_crossover_curve: pd.DataFrame
    trade_log: pd.DataFrame


def evaluate_policy_model(
    model,
    frame: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
) -> EvaluationSummary:
    return evaluate_policy_run(
        model=model,
        frame=frame,
        feature_config=feature_config,
        env_config=env_config,
    ).summary


def evaluate_policy_run(
    model,
    frame: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
) -> EvaluationArtifacts:
    indicator_config = feature_config or FeatureConfig()
    market_config = env_config or EnvConfig()
    env = TradingEnv(frame=frame, feature_config=indicator_config, env_config=market_config)

    observation, _ = env.reset()
    terminated = False
    portfolio_rows = [
        {
            "date": env.frame.loc[0, "date"],
            "portfolio_value": env.state.portfolio_value,
        }
    ]
    trade_rows: list[dict[str, object]] = []

    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, _, info = env.step(int(action))
        portfolio_rows.append(
            {
                "date": info["date"],
                "portfolio_value": info["portfolio_value"],
            }
        )

        if info["executed_trade"]:
            trade_rows.append(
                {
                    "date": info["date"],
                    "action": action_name(int(action)),
                    "execution_price": info["execution_price"],
                    "mark_price": info["mark_price"],
                    "portfolio_value": info["portfolio_value"],
                    "cash": info["cash"],
                    "shares": info["shares"],
                    "position_after": position_name(int(info["position"])),
                }
            )

    portfolio_curve = pd.DataFrame(portfolio_rows)
    trade_log = pd.DataFrame(trade_rows)
    benchmark_curve = simulate_buy_and_hold_curve(
        frame=frame,
        initial_cash=market_config.initial_cash,
        fee_rate=market_config.fee_rate,
    )
    ma_crossover_curve = simulate_ma_crossover_curve(
        frame=frame,
        initial_cash=market_config.initial_cash,
        fee_rate=market_config.fee_rate,
    )

    summary = summarize_portfolio(
        portfolio_values=portfolio_curve["portfolio_value"].to_numpy(dtype=float),
        trade_count=env.state.trade_count,
        initial_cash=market_config.initial_cash,
        benchmark_final_value=float(benchmark_curve["portfolio_value"].iloc[-1]),
        benchmark_return=compute_total_return(benchmark_curve["portfolio_value"], market_config.initial_cash),
        ma_crossover_final_value=float(ma_crossover_curve["portfolio_value"].iloc[-1]),
        ma_crossover_return=compute_total_return(
            ma_crossover_curve["portfolio_value"],
            market_config.initial_cash,
        ),
    )
    return EvaluationArtifacts(
        summary=summary,
        portfolio_curve=portfolio_curve,
        benchmark_curve=benchmark_curve,
        ma_crossover_curve=ma_crossover_curve,
        trade_log=trade_log,
    )


def summarize_portfolio(
    portfolio_values: np.ndarray,
    trade_count: int,
    initial_cash: float,
    benchmark_final_value: float,
    benchmark_return: float,
    ma_crossover_final_value: float,
    ma_crossover_return: float,
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
        ma_crossover_return=float(ma_crossover_return),
        ma_crossover_final_value=float(ma_crossover_final_value),
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


def compute_total_return(portfolio_values: pd.Series, initial_cash: float) -> float:
    return float((float(portfolio_values.iloc[-1]) / initial_cash) - 1)


def simulate_buy_and_hold_curve(
    frame: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
) -> pd.DataFrame:
    rows = [{"date": frame.loc[0, "date"], "portfolio_value": initial_cash}]

    first_execution_price = float(frame.loc[1, "open"])
    spendable_cash = initial_cash / (1 + fee_rate)
    shares = spendable_cash / first_execution_price

    for index in range(1, len(frame)):
        rows.append(
            {
                "date": frame.loc[index, "date"],
                "portfolio_value": shares * float(frame.loc[index, "close"]),
            }
        )

    return pd.DataFrame(rows)


def simulate_ma_crossover_curve(
    frame: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
) -> pd.DataFrame:
    state = make_initial_state(initial_cash)
    rows = [{"date": frame.loc[0, "date"], "portfolio_value": state.portfolio_value}]

    for index in range(len(frame) - 1):
        bullish = bool(frame.loc[index, "sma_10"] > frame.loc[index, "sma_50"])

        if bullish and state.position == 0:
            action = ACTION_BUY
        elif not bullish and state.position == POSITION_LONG:
            action = ACTION_SELL
        else:
            action = ACTION_HOLD

        next_index = index + 1
        state = apply_action(
            state=state,
            action=action,
            execution_price=float(frame.loc[next_index, "open"]),
            mark_price=float(frame.loc[next_index, "close"]),
            fee_rate=fee_rate,
        )
        rows.append(
            {
                "date": frame.loc[next_index, "date"],
                "portfolio_value": state.portfolio_value,
            }
        )

    return pd.DataFrame(rows)


def save_trade_log(trade_log: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_log.to_csv(path, index=False)


def save_performance_plot(
    artifacts: EvaluationArtifacts,
    output_path: str | Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. Install project dependencies first."
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(
        artifacts.portfolio_curve["date"],
        artifacts.portfolio_curve["portfolio_value"],
        label="RL agent",
        linewidth=2,
    )
    plt.plot(
        artifacts.benchmark_curve["date"],
        artifacts.benchmark_curve["portfolio_value"],
        label="Buy and hold",
        linewidth=2,
    )
    plt.plot(
        artifacts.ma_crossover_curve["date"],
        artifacts.ma_crossover_curve["portfolio_value"],
        label="MA crossover",
        linewidth=2,
    )
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate_saved_model(
    model_path: str,
    data_config: DataConfig | None = None,
    feature_config: FeatureConfig | None = None,
    env_config: EnvConfig | None = None,
) -> EvaluationArtifacts:
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
    return evaluate_policy_run(
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
    parser.add_argument("--plot-out", type=str, default="artifacts/performance.png")
    parser.add_argument("--trade-log-out", type=str, default="artifacts/trade_log.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_config = DataConfig(ticker=args.ticker, start=args.start, end=args.end)
    artifacts = evaluate_saved_model(
        model_path=args.model_path,
        data_config=data_config,
    )
    save_performance_plot(artifacts, args.plot_out)
    save_trade_log(artifacts.trade_log, args.trade_log_out)
    print(artifacts.summary.to_report_text(prefix="Test"))
    print(f"  plot_saved_to: {args.plot_out}")
    print(f"  trade_log_saved_to: {args.trade_log_out}")


if __name__ == "__main__":
    main()
