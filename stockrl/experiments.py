from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stockrl.config import DataConfig, EnvConfig, FeatureConfig, TrainConfig
from stockrl.evaluate import (
    EvaluationArtifacts,
    save_performance_plot,
    save_trade_log,
)
from stockrl.train import prepare_frames, train_agent_on_frames


def run_seed_experiments(
    seeds: list[int],
    data_config: DataConfig,
    feature_config: FeatureConfig,
    env_config: EnvConfig,
    base_train_config: TrainConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_frame, validation_frame, test_frame = prepare_frames(
        data_config=data_config,
        feature_config=feature_config,
    )

    results: list[dict[str, float | int | str]] = []
    best_run: tuple[int, EvaluationArtifacts] | None = None
    best_score = float("-inf")

    for seed in seeds:
        model_path = output_path / f"ppo_seed_{seed}.zip"
        train_config = TrainConfig(
            total_timesteps=base_train_config.total_timesteps,
            random_seed=seed,
            learning_rate=base_train_config.learning_rate,
            n_steps=base_train_config.n_steps,
            batch_size=base_train_config.batch_size,
            ent_coef=base_train_config.ent_coef,
        )

        model, validation_artifacts = train_agent_on_frames(
            train_frame=train_frame,
            validation_frame=validation_frame,
            feature_config=feature_config,
            env_config=env_config,
            train_config=train_config,
            model_out=model_path,
        )
        test_artifacts = evaluate_on_test_frame(
            model=model,
            test_frame=test_frame,
            feature_config=feature_config,
            env_config=env_config,
        )

        results.append(
            {
                "seed": seed,
                "timesteps": train_config.total_timesteps,
                "validation_total_return": validation_artifacts.summary.total_return,
                "validation_sharpe": validation_artifacts.summary.sharpe_ratio,
                "test_total_return": test_artifacts.summary.total_return,
                "test_sharpe": test_artifacts.summary.sharpe_ratio,
                "test_max_drawdown": test_artifacts.summary.max_drawdown,
                "test_trade_count": test_artifacts.summary.trade_count,
                "buy_and_hold_return": test_artifacts.summary.benchmark_return,
                "ma_crossover_return": test_artifacts.summary.ma_crossover_return,
                "model_path": str(model_path),
            }
        )

        if validation_artifacts.summary.total_return > best_score:
            best_score = validation_artifacts.summary.total_return
            best_run = (seed, test_artifacts)

    results_frame = pd.DataFrame(results).sort_values(
        by=["test_total_return", "test_sharpe"],
        ascending=False,
    )
    results_frame.to_csv(output_path / "experiment_results.csv", index=False)

    if best_run is not None:
        best_seed, best_artifacts = best_run
        save_performance_plot(best_artifacts, output_path / f"best_seed_{best_seed}_performance.png")
        save_trade_log(best_artifacts.trade_log, output_path / f"best_seed_{best_seed}_trade_log.csv")

    return results_frame


def evaluate_on_test_frame(
    model,
    test_frame: pd.DataFrame,
    feature_config: FeatureConfig,
    env_config: EnvConfig,
) -> EvaluationArtifacts:
    from stockrl.evaluate import evaluate_policy_run

    return evaluate_policy_run(
        model=model,
        frame=test_frame,
        feature_config=feature_config,
        env_config=env_config,
    )


def parse_seeds(seeds_text: str) -> list[int]:
    seeds: list[int] = []
    for part in seeds_text.split(","):
        piece = part.strip()
        if not piece:
            continue
        seeds.append(int(piece))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-seed PPO experiments on SPY.")
    parser.add_argument("--ticker", type=str, default="SPY")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seeds", type=str, default="1,7,42,123")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = run_seed_experiments(
        seeds=parse_seeds(args.seeds),
        data_config=DataConfig(ticker=args.ticker, start=args.start, end=args.end),
        feature_config=FeatureConfig(),
        env_config=EnvConfig(),
        base_train_config=TrainConfig(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
        ),
        output_dir=args.output_dir,
    )
    print(results.to_string(index=False))
    print(f"results_saved_to: {Path(args.output_dir) / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
