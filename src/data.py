from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
SYMBOL_CANDIDATES = ("symbol", "ticker")


def load_stock_data(
    csv_path: str | Path,
    *,
    default_symbol: str | None = None,
    drop_invalid_dates: bool = True,
) -> pd.DataFrame:
    """
    Load raw stock data from CSV and apply lightweight normalization.

    Normalization performed:
    - lowercase and strip column names,
    - normalize symbol column to `symbol`,
    - parse `date` as datetime,
    - coerce OHLCV columns to numeric where possible,
    - strip symbol values,
    - sort by `symbol` and `date`.

    This function does not automatically deduplicate rows or remove rows with
    missing OHLCV values, so that validation code can inspect raw data quality.
    """
    data = pd.read_csv(csv_path)
    data = data.copy()

    data.columns = _normalize_column_names(data.columns)
    _validate_required_columns(data.columns)

    symbol_column = _find_symbol_column(data.columns)
    if symbol_column is None:
        if default_symbol is None:
            raise ValueError(
                "No symbol column found. Expected one of "
                f"{list(SYMBOL_CANDIDATES)} or provide default_symbol."
            )
        data["symbol"] = default_symbol
    elif symbol_column != "symbol":
        data = data.rename(columns={symbol_column: "symbol"})

    data["symbol"] = data["symbol"].astype(str).str.strip()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if drop_invalid_dates:
        data = data.dropna(subset=["date"])

    data = data.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    return data


def select_symbol(data: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    """Return a copy filtered to one symbol, or the full dataset if symbol is None."""
    if symbol is None:
        return data.copy()

    normalized_symbol = symbol.strip()
    filtered = data.loc[data["symbol"] == normalized_symbol].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for symbol '{symbol}'")
    return filtered


def _normalize_column_names(columns: Iterable[str]) -> list[str]:
    return [str(column).strip().lower() for column in columns]


def _validate_required_columns(columns: Iterable[str]) -> None:
    column_set = set(columns)
    missing = REQUIRED_COLUMNS.difference(column_set)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _find_symbol_column(columns: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in SYMBOL_CANDIDATES:
        if candidate in column_set:
            return candidate
    return None