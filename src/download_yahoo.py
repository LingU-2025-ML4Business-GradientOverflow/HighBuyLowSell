from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DEFAULT_START_DATE, DEFAULT_TICKERS, RAW_DATA_COLUMNS


REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close"]
OPTIONAL_COLUMNS = [col for col in RAW_DATA_COLUMNS if col not in REQUIRED_COLUMNS]


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


def validate_date(date_str: str | None, arg_name: str) -> str | None:
    if date_str is None:
        return None
    try:
        return pd.Timestamp(date_str).strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"Invalid {arg_name}: {date_str!r}. Expected YYYY-MM-DD.") from exc


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    return result


def reshape_downloaded_data(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Convert yfinance download output into a normalized flat DataFrame.
    Supports both single-ticker and multi-ticker responses.
    """
    if downloaded is None or downloaded.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    if isinstance(downloaded.columns, pd.MultiIndex):
        frames: list[pd.DataFrame] = []

        available_tickers = set(downloaded.columns.get_level_values(0))
        for ticker in tickers:
            if ticker not in available_tickers:
                continue

            ticker_frame = downloaded[ticker].copy().reset_index()
            ticker_frame = normalize_columns(ticker_frame)
            ticker_frame["symbol"] = ticker
            frames.append(ticker_frame)

        if not frames:
            raise ValueError("Download returned data, but none of the requested tickers were found.")

        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = downloaded.reset_index()
        combined = normalize_columns(combined)
        combined["symbol"] = tickers[0]

    return combined


def select_and_validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    # Compatibility fallback in case the source provides adj close instead of close.
    if "close" not in normalized.columns and "adj close" in normalized.columns:
        normalized = normalized.rename(columns={"adj close": "close"})

    missing_required = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns after normalization: {missing_required}. "
            f"Available columns: {sorted(normalized.columns.tolist())}"
        )

    selected_columns = [col for col in RAW_DATA_COLUMNS if col in normalized.columns]
    result = normalized[selected_columns].copy()

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)

    before = len(result)
    result = result.dropna(subset=["date", "open", "high", "low", "close"])
    dropped = before - len(result)

    if dropped > 0:
        print(f"Warning: dropped {dropped} rows with incomplete OHLC data.", file=sys.stderr)

    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    return result


def download_daily_prices(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    start = validate_date(start, "--start")
    end = validate_date(end, "--end")

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

    reshaped = reshape_downloaded_data(downloaded, tickers)
    normalized = select_and_validate_columns(reshaped)

    if normalized.empty:
        raise ValueError("Downloaded dataset is empty after cleaning.")

    return normalized


def main() -> None:
    args = parse_args()

    try:
        dataset = download_daily_prices(args.tickers, args.start, args.end)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)

        print(f"Saved {len(dataset)} rows to {output_path}")
        print(f"Tickers: {', '.join(sorted(dataset['symbol'].unique()))}")
        print(f"Date range: {dataset['date'].min().date()} -> {dataset['date'].max().date()}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()