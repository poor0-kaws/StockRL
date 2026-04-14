from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def download_price_data(
    ticker: str = "SPY",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily price data with yfinance."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for downloading data. Install project dependencies first."
        ) from exc

    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if frame.empty:
        raise ValueError(f"No data returned for ticker={ticker!r}.")

    frame = frame.reset_index()
    return normalize_price_frame(frame)


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, sort by date, and reject broken inputs."""
    if frame.empty:
        raise ValueError("Price data is empty.")

    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}.")

    normalized = normalized.loc[:, list(REQUIRED_COLUMNS)].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], utc=False)
    normalized = normalized.sort_values("date").reset_index(drop=True)

    if normalized["date"].duplicated().any():
        raise ValueError("Price data contains duplicate dates.")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if normalized[numeric_columns].isna().any().any():
        raise ValueError("Price data contains missing numeric values.")

    return normalized


def split_by_time(
    frame: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> DatasetSplits:
    """Split the frame in time order without overlap."""
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    total_rows = len(frame)
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    if train_end < 2:
        raise ValueError("Not enough rows for a train split.")

    if validation_end <= train_end:
        raise ValueError("Validation split is too small.")

    if validation_end >= total_rows:
        raise ValueError("Test split is too small.")

    train = frame.iloc[:train_end].reset_index(drop=True)
    validation = frame.iloc[train_end:validation_end].reset_index(drop=True)
    test = frame.iloc[validation_end:].reset_index(drop=True)

    return DatasetSplits(train=train, validation=validation, test=test)
