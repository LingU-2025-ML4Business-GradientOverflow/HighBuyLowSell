from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import DEFAULT_START_DATE, DEFAULT_TICKERS, RAW_DATA_COLUMNS
from data import load_stock_data


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
    return parser.parse_args()


def build_summary(data: pd.DataFrame) -> dict:
    per_symbol = []
    for symbol, frame in data.groupby("symbol"):
        frame = frame.sort_values("date")
        per_symbol.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start_date": frame["date"].min().date().isoformat(),
                "end_date": frame["date"].max().date().isoformat(),
                "missing_close": int(frame["close"].isna().sum()),
                "missing_volume": int(frame["volume"].isna().sum()),
            }
        )

    summary = {
        "expected_tickers": DEFAULT_TICKERS,
        "expected_start_date": DEFAULT_START_DATE,
        "expected_columns": RAW_DATA_COLUMNS,
        "row_count": int(len(data)),
        "ticker_count": int(data["symbol"].nunique()),
        "tickers": sorted(data["symbol"].unique().tolist()),
        "global_start_date": data["date"].min().date().isoformat(),
        "global_end_date": data["date"].max().date().isoformat(),
        "duplicate_symbol_date_rows": int(data.duplicated(subset=["symbol", "date"]).sum()),
        "per_symbol": per_symbol,
    }
    return summary


def main() -> None:
    args = parse_args()
    dataset = load_stock_data(args.input)
    summary = build_summary(dataset)

    output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved raw data summary to {output_path}")


if __name__ == "__main__":
    main()
