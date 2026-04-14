from __future__ import annotations

import pandas as pd
import pytest

from stockrl.config import DataConfig, EnvConfig, FeatureConfig, TrainConfig
from stockrl.data_loader import normalize_price_frame
from stockrl.train import train_agent


def make_download_frame():
    periods = 120
    dates = pd.date_range("2022-01-01", periods=periods, freq="D")
    close = pd.Series(100 + (pd.Series(range(periods)) * 0.2), dtype=float)
    frame = pd.DataFrame(
        {
            "Date": dates,
            "Open": close + 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": 1_000,
        }
    )
    return normalize_price_frame(frame)


@pytest.mark.smoke
def test_train_agent_smoke(monkeypatch, tmp_path) -> None:
    stable_baselines = pytest.importorskip("stable_baselines3")
    assert stable_baselines is not None

    def fake_download_price_data(ticker: str, start: str | None, end: str | None):
        assert ticker == "SPY"
        return make_download_frame()

    monkeypatch.setattr("stockrl.train.download_price_data", fake_download_price_data)

    _, summary = train_agent(
        data_config=DataConfig(train_ratio=0.6, validation_ratio=0.2),
        feature_config=FeatureConfig(scale_features=True),
        env_config=EnvConfig(initial_cash=1_000.0, fee_rate=0.0),
        train_config=TrainConfig(total_timesteps=64, random_seed=1),
        model_out=tmp_path / "ppo_spy.zip",
    )

    assert summary.final_portfolio_value > 0
