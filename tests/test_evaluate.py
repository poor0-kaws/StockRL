from __future__ import annotations

import pandas as pd
import pytest

from stockrl.evaluate import compute_buy_and_hold_value, summarize_portfolio


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
    )

    assert summary.total_return == pytest.approx(0.10)
    assert summary.max_drawdown == pytest.approx(-0.25)
    assert summary.trade_count == 2
