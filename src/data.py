from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
SYMBOL_CANDIDATES = ("symbol", "name")


def load_stock_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw stock data and normalize common column names."""
    data = pd.read_csv(csv_path)
    data.columns = [column.strip().lower() for column in data.columns]

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    symbol_column = _find_symbol_column(data.columns)
    if symbol_column is None:
        data["symbol"] = "UNKNOWN"
    elif symbol_column != "symbol":
        data = data.rename(columns={symbol_column: "symbol"})

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "close", "volume"])
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    return data


def select_symbol(data: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    """Filter a dataset to a single symbol when requested."""
    if symbol is None:
        return data.copy()

    filtered = data.loc[data["symbol"] == symbol].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for symbol '{symbol}'")
    return filtered


def _find_symbol_column(columns: list[str] | pd.Index) -> str | None:
    for candidate in SYMBOL_CANDIDATES:
        if candidate in columns:
            return candidate
    return None
