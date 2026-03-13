from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DEFAULT_START_DATE, DEFAULT_TICKERS, RAW_DATA_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download daily stock data from Yahoo Finance.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker list to download.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Exclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/yahoo_daily_prices.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def download_daily_prices(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    downloaded = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if downloaded is None or downloaded.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        frames = []
        for ticker in tickers:
            if ticker not in downloaded.columns.get_level_values(0):
                continue

            ticker_frame = downloaded[ticker].copy()
            ticker_frame = ticker_frame.reset_index()
            ticker_frame.columns = [str(column).strip().lower() for column in ticker_frame.columns]
            ticker_frame["symbol"] = ticker
            frames.append(ticker_frame)

        if not frames:
            raise ValueError("Download returned data, but none of the requested tickers were found.")

        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = downloaded.reset_index()
        combined.columns = [str(column).strip().lower() for column in combined.columns]
        combined["symbol"] = tickers[0]

    normalized = combined.rename(columns={"adj close": "close"})
    available = [column for column in RAW_DATA_COLUMNS if column in normalized.columns]
    missing = set(RAW_DATA_COLUMNS).difference(available)
    if missing:
        raise ValueError(f"Missing expected output columns after normalization: {sorted(missing)}")

    normalized = normalized[RAW_DATA_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    # Multi-market downloads can include non-trading dates for a subset of symbols.
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized.sort_values(["symbol", "date"]).reset_index(drop=True)
    return normalized


def main() -> None:
    args = parse_args()
    dataset = download_daily_prices(args.tickers, args.start, args.end)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Saved {len(dataset)} rows to {output_path}")
    print(f"Tickers: {', '.join(sorted(dataset['symbol'].unique()))}")
    print(f"Date range: {dataset['date'].min().date()} -> {dataset['date'].max().date()}")


if __name__ == "__main__":
    main()
