from __future__ import annotations

import pandas as pd
import pytest

from stockrl.data_loader import normalize_price_frame, split_by_time


def test_normalize_price_frame_rejects_duplicate_dates() -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Volume": [100, 200],
        }
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        normalize_price_frame(frame)


def test_split_by_time_keeps_strict_order_with_no_overlap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": range(10),
        }
    )

    splits = split_by_time(frame, train_ratio=0.6, validation_ratio=0.2)

    assert len(splits.train) == 6
    assert len(splits.validation) == 2
    assert len(splits.test) == 2
    assert splits.train["date"].max() < splits.validation["date"].min()
    assert splits.validation["date"].max() < splits.test["date"].min()
