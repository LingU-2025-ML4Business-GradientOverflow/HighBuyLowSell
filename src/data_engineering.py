import os
from datetime import datetime
import pandas as pd
import yfinance as yf

# =========================================================
# Part 1: Configuration Parameters
# =========================================================
# Target stock tickers (removed AAPL, keep 6 specified stocks)
TICKERS = ["NVDA", "MSFT", "GOOGL", "BABA", "BIDU", "0700.HK"]
START_DATE = "2023-01-01"
END_DATE = "2025-01-01"
DATA_DIR = "data"
LOG_DIR = "logs"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
CLEANED_PARQUET_PATH = os.path.join(DATA_DIR, "cleaned_data.parquet")
LOG_PATH = os.path.join(LOG_DIR, "data_cleaning_log.md")


# =========================================================
# Part 2: Create Output Directories
# =========================================================
def ensure_directories():
    """Create data and logs directories if they do not exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# Part 3: Download Single Stock Data from Yahoo Finance
# =========================================================
def download_stock_data(ticker, start_date, end_date):
    """Download daily OHLCV data for a single ticker
    Args:
        ticker: Stock symbol
        start_date: Start date of data
        end_date: End date of data
    Returns:
        DataFrame with raw stock data
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )
    if df.empty:
        raise ValueError(f"No data downloaded for ticker={ticker}. Please check the ticker or date range.")
    return df


# =========================================================
# Part 4: Download Multiple Stocks Data
# =========================================================
def download_multiple_stock_data(tickers, start_date, end_date):
    """Download data for all tickers and store in a list
    Args:
        tickers: List of stock symbols
        start_date: Start date of data
        end_date: End date of data
    Returns:
        List of DataFrames for each stock
    """
    all_dfs = []
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = download_stock_data(ticker, start_date, end_date)
        all_dfs.append(df)
        print(f"Successfully downloaded {len(df)} rows for {ticker}")
    return all_dfs


# =========================================================
# Part 5: Standardize Raw Data Format
# =========================================================
def standardize_raw_data(df, ticker):
    """Standardize column names, data types and add ticker column
    Args:
        df: Raw stock data
        ticker: Stock symbol
    Returns:
        Standardized DataFrame
    """
    df = df.copy()
    # Handle multi-index columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Convert index to date column
    df = df.reset_index()
    # Rename columns to unified lowercase format
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }
    df = df.rename(columns=rename_map)
    # Validate required columns
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing_required_cols = [col for col in required_cols if col not in df.columns]
    if missing_required_cols:
        raise ValueError(f"Missing required columns after renaming for {ticker}: {missing_required_cols}")
    df = df[required_cols]
    # Add stock identifier column
    df["ticker"] = ticker
    # Convert date format and sort data
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================================================
# Part 6: Clean Stock Data (Remove Holidays/Zero Volume Rows)
# =========================================================
def clean_stock_data(df):
    """Clean data: remove duplicates, invalid values, holidays (volume=0)
    Args:
        df: Merged standardized raw data
    Returns:
        Cleaned DataFrame and cleaning summary statistics
    """
    df = df.copy()
    summary = {}
    # Record initial data size
    initial_rows = len(df)
    summary["initial_rows"] = initial_rows
    summary["missing_before"] = df.isna().sum().to_dict()

    # 1. Remove duplicate rows (date + ticker combination)
    duplicated_pairs = int(df.duplicated(subset=["date", "ticker"]).sum())
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first").copy()

    # 2. Remove rows with invalid OHLC price relationships
    invalid_ohlc_mask = (
            (df["high"] < df["low"]) |
            (df["open"] > df["high"]) |
            (df["open"] < df["low"]) |
            (df["close"] > df["high"]) |
            (df["close"] < df["low"])
    )
    invalid_ohlc_rows = int(invalid_ohlc_mask.sum())
    df = df.loc[~invalid_ohlc_mask].copy()

    # 3. Forward fill missing price values (group by ticker to avoid cross-stock filling)
    price_cols = ["open", "high", "low", "close", "adj_close"]
    summary["missing_before_ffill_price_cols"] = df[price_cols].isna().sum().to_dict()
    df[price_cols] = df.groupby("ticker")[price_cols].ffill()

    # 4. Fill missing volume values with 0
    missing_volume_before_fill = int(df["volume"].isna().sum())
    df["volume"] = df["volume"].fillna(0)

    # 5. Remove ALL rows with zero volume (holidays, half-trading days, market closures)
    zero_volume_rows = int((df["volume"] == 0).sum())
    df = df.loc[df["volume"] > 0].copy()

    # 6. Remove rows with missing critical fields
    critical_cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]
    rows_before_dropna = len(df)
    df = df.dropna(subset=critical_cols).copy()
    rows_dropped_after_dropna = int(rows_before_dropna - len(df))

    # 7. Remove rows with non-positive prices or negative volume
    invalid_value_mask = (
            (df["open"] <= 0) |
            (df["high"] <= 0) |
            (df["low"] <= 0) |
            (df["close"] <= 0) |
            (df["adj_close"] <= 0) |
            (df["volume"] < 0)
    )
    invalid_value_rows = int(invalid_value_mask.sum())
    df = df.loc[~invalid_value_mask].copy()

    # Sort data by ticker and date
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Generate cleaning summary statistics
    summary["final_rows"] = len(df)
    summary["dropped_rows_total"] = int(initial_rows - len(df))
    summary["duplicated_date_ticker_pairs_removed"] = duplicated_pairs
    summary["invalid_ohlc_rows_removed"] = invalid_ohlc_rows
    summary["missing_volume_before_fill"] = missing_volume_before_fill
    summary["zero_volume_holiday_rows_removed"] = zero_volume_rows
    summary["rows_dropped_after_dropna"] = rows_dropped_after_dropna
    summary["invalid_value_rows_removed"] = invalid_value_rows
    summary["missing_after"] = df.isna().sum().to_dict()
    summary["date_min"] = df["date"].min().strftime("%Y-%m-%d") if not df.empty else None
    summary["date_max"] = df["date"].max().strftime("%Y-%m-%d") if not df.empty else None
    summary["unique_tickers"] = df["ticker"].nunique()
    summary["ticker_list"] = sorted(df["ticker"].unique().tolist())

    return df, summary


# =========================================================
# Part 7: Save Data Cleaning Log (Markdown Format)
# =========================================================
def save_cleaning_log(summary, tickers, start_date, end_date, log_path):
    """Save detailed cleaning report to markdown file"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tickers_md = "`, `".join(tickers)
    valid_tickers_md = "`, `".join(summary["ticker_list"])

    log_text = f"""# Data Cleaning Log
## Metadata
- Tickers requested: `{tickers_md}`
- Tickers with valid data: `{valid_tickers_md}`
- Source: `Yahoo Finance`
- Requested period: `{start_date}` to `{end_date}`
- Log generated at: `{now_str}`

## Task Scope
This file documents the data engineering stage of the stock machine learning project.

## Cleaning Objectives
- Download daily OHLCV stock data for multiple tickers
- **Remove holiday/half-trading day data (volume=0)**
- Standardize data schema for downstream EDA, labeling and feature engineering
- Merge all ticker data into a single unified DataFrame

## Summary
- Initial total rows (all tickers): **{summary["initial_rows"]}**
- Final total rows (cleaned): **{summary["final_rows"]}**
- Total dropped rows: **{summary["dropped_rows_total"]}**
- Date range after cleaning: **{summary["date_min"]}** to **{summary["date_max"]}**

## Detailed Cleaning Actions
- Duplicated (date, ticker) pairs removed: **{summary["duplicated_date_ticker_pairs_removed"]}**
- Invalid OHLC rows removed: **{summary["invalid_ohlc_rows_removed"]}**
- Zero-volume holiday/half-day rows removed: **{summary["zero_volume_holiday_rows_removed"]}**
- Missing volume values before fill: **{summary["missing_volume_before_fill"]}**
- Rows dropped after critical null check: **{summary["rows_dropped_after_dropna"]}**
- Invalid value rows removed: **{summary["invalid_value_rows_removed"]}**

## Output Files
- `data/raw_data.csv`: Raw merged data for all tickers
- `data/cleaned_data.csv`: Cleaned data (no holidays/zero volume)
- `data/cleaned_data.parquet`: Cleaned data for ML pipeline
- `logs/data_cleaning_log.md`: Data cleaning documentation
"""
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)


# =========================================================
# Part 8: Main Data Pipeline
# =========================================================
def main():
    """Main execution function for the data engineering pipeline"""
    print("Starting data engineering pipeline for multiple tickers...")
    # Create directories
    ensure_directories()
    # Download data for all stocks
    raw_dfs_list = download_multiple_stock_data(TICKERS, START_DATE, END_DATE)
    # Standardize and merge data
    raw_standardized_dfs = []
    for ticker, df in zip(TICKERS, raw_dfs_list):
        std_df = standardize_raw_data(df, ticker)
        raw_standardized_dfs.append(std_df)
    raw_df = pd.concat(raw_standardized_dfs, ignore_index=True)
    print(f"Standardized merged raw data shape: {raw_df.shape}")
    # Save raw data
    raw_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw merged data saved to: {RAW_DATA_PATH}")

    # Clean data
    cleaned_df, summary = clean_stock_data(raw_df)
    print(f"Cleaned merged data shape: {cleaned_df.shape}")
    print(f"Removed {summary['zero_volume_holiday_rows_removed']} zero-volume holiday/half-day rows")
    print(f"Valid unique tickers: {summary['unique_tickers']} ({summary['ticker_list']})")

    # Save cleaned data
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
    # cleaned_df.to_parquet(CLEANED_PARQUET_PATH, index=False)
    # Save log file
    save_cleaning_log(summary, TICKERS, START_DATE, END_DATE, LOG_PATH)
    print("Pipeline completed successfully!")


# =========================================================
# Script Entry Point
# =========================================================
if __name__ == "__main__":
    main()