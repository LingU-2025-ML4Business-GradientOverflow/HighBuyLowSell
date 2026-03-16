from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEFAULT_START_DATE, DEFAULT_TICKERS, RAW_DATA_COLUMNS
from data import load_stock_data


REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = [col for col in RAW_DATA_COLUMNS if col not in REQUIRED_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize raw stock data.")
    parser.add_argument(
        "--input",
        default="data/raw/yahoo_daily_prices.csv",
        help="Path to raw CSV file.",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/raw_data_summary.json",
        help="Path to save validation summary JSON.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with non-zero status code if warnings are found.",
    )
    return parser.parse_args()


def safe_iso_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def build_summary(data: pd.DataFrame) -> dict[str, Any]:
    per_symbol: list[dict[str, Any]] = []

    if data.empty:
        return {
            "expected_tickers": DEFAULT_TICKERS,
            "expected_start_date": DEFAULT_START_DATE,
            "expected_columns": RAW_DATA_COLUMNS,
            "row_count": 0,
            "ticker_count": 0,
            "tickers": [],
            "global_start_date": None,
            "global_end_date": None,
            "duplicate_symbol_date_rows": 0,
            "missing_expected_tickers": sorted(DEFAULT_TICKERS),
            "unexpected_tickers": [],
            "per_symbol": [],
        }

    for symbol, frame in data.groupby("symbol", sort=True):
        frame = frame.sort_values("date")
        per_symbol.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "unique_dates": int(frame["date"].nunique()),
                "start_date": safe_iso_date(frame["date"].min()),
                "end_date": safe_iso_date(frame["date"].max()),
                "missing_close": int(frame["close"].isna().sum()) if "close" in frame.columns else None,
                "missing_volume": int(frame["volume"].isna().sum()) if "volume" in frame.columns else None,
                "duplicate_symbol_date_rows": int(frame.duplicated(subset=["symbol", "date"]).sum()),
            }
        )

    actual_tickers = sorted(data["symbol"].dropna().unique().tolist()) if "symbol" in data.columns else []
    expected_tickers = sorted(DEFAULT_TICKERS)

    summary = {
        "expected_tickers": expected_tickers,
        "expected_start_date": DEFAULT_START_DATE,
        "expected_columns": RAW_DATA_COLUMNS,
        "row_count": int(len(data)),
        "ticker_count": int(data["symbol"].nunique()) if "symbol" in data.columns else 0,
        "tickers": actual_tickers,
        "global_start_date": safe_iso_date(data["date"].min()) if "date" in data.columns else None,
        "global_end_date": safe_iso_date(data["date"].max()) if "date" in data.columns else None,
        "duplicate_symbol_date_rows": int(data.duplicated(subset=["symbol", "date"]).sum())
        if {"symbol", "date"}.issubset(data.columns)
        else None,
        "missing_expected_tickers": sorted(set(expected_tickers) - set(actual_tickers)),
        "unexpected_tickers": sorted(set(actual_tickers) - set(expected_tickers)),
        "per_symbol": per_symbol,
    }
    return summary


def validate_data(data: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    missing_expected = summary.get("missing_expected_tickers", [])
    if missing_expected:
        warnings.append(f"Missing expected tickers: {missing_expected}")

    unexpected_tickers = summary.get("unexpected_tickers", [])
    if unexpected_tickers:
        warnings.append(f"Unexpected tickers present: {unexpected_tickers}")

    duplicate_rows = summary.get("duplicate_symbol_date_rows")
    if isinstance(duplicate_rows, int) and duplicate_rows > 0:
        errors.append(f"Found {duplicate_rows} duplicate rows on ['symbol', 'date'].")

    if data.empty:
        errors.append("Dataset is empty.")

    if "date" in data.columns and not data.empty:
        global_start = pd.Timestamp(summary["global_start_date"]) if summary.get("global_start_date") else None
        expected_start = pd.Timestamp(DEFAULT_START_DATE)

        if global_start is not None and global_start > expected_start:
            warnings.append(
                f"Global start date {global_start.date().isoformat()} is later than expected "
                f"{expected_start.date().isoformat()}."
            )

    if "close" in data.columns:
        missing_close = int(data["close"].isna().sum())
        if missing_close > 0:
            warnings.append(f"Found {missing_close} rows with missing close values.")

    if "symbol" in data.columns:
        invalid_symbol_mask = data["symbol"].isna() | data["symbol"].astype("string").str.strip().eq("")
        missing_symbol = int(invalid_symbol_mask.sum())
        if missing_symbol > 0:
            errors.append(f"Found {missing_symbol} rows with missing symbol values.")

    if "date" in data.columns:
        missing_date = int(data["date"].isna().sum())
        if missing_date > 0:
            errors.append(f"Found {missing_date} rows with missing date values.")

    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    summary["status"] = status
    summary["errors"] = errors
    summary["warnings"] = warnings
    return summary


def main() -> None:
    args = parse_args()

    try:
        dataset = load_stock_data(args.input)
        summary = build_summary(dataset)
        summary = validate_data(dataset, summary)

        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Saved raw data summary to {output_path}")

        if summary["status"] == "error":
            raise SystemExit(1)
        if args.fail_on_warning and summary["status"] == "warning":
            raise SystemExit(2)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
