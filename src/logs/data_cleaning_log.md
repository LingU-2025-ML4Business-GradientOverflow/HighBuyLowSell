# Data Cleaning Log
## Metadata
- Tickers requested: `NVDA`, `MSFT`, `GOOGL`, `BABA`, `BIDU`, `0700.HK`
- Tickers with valid data: `0700.HK`, `BABA`, `BIDU`, `GOOGL`, `MSFT`, `NVDA`
- Source: `Yahoo Finance`
- Requested period: `2023-01-01` to `2025-01-01`
- Log generated at: `2026-03-13 22:29:38`

## Task Scope
This file documents the data engineering stage of the stock machine learning project.

## Cleaning Objectives
- Download daily OHLCV stock data for multiple tickers
- **Remove holiday/half-trading day data (volume=0)**
- Standardize data schema for downstream EDA, labeling and feature engineering
- Merge all ticker data into a single unified DataFrame

## Summary
- Initial total rows (all tickers): **2999**
- Final total rows (cleaned): **2997**
- Total dropped rows: **2**
- Date range after cleaning: **2023-01-03** to **2024-12-31**

## Detailed Cleaning Actions
- Duplicated (date, ticker) pairs removed: **0**
- Invalid OHLC rows removed: **0**
- Zero-volume holiday/half-day rows removed: **2**
- Missing volume values before fill: **0**
- Rows dropped after critical null check: **0**
- Invalid value rows removed: **0**

## Output Files
- `data/raw_data.csv`: Raw merged data for all tickers
- `data/cleaned_data.csv`: Cleaned data (no holidays/zero volume)
- `data/cleaned_data.parquet`: Cleaned data for ML pipeline
- `logs/data_cleaning_log.md`: Data cleaning documentation
