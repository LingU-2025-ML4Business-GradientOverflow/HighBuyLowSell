# HighBuyLowSell

Traditional machine learning group project for stock movement prediction using engineered time-series features, Yahoo Finance data, baseline models, and simple evaluation.

## Project Goal

Build a reproducible pipeline for:

1. downloading and validating historical stock data,
2. defining a next-day direction prediction task,
3. engineering baseline time-series features,
4. training and comparing classical machine learning models, and
5. preparing the project for later evaluation and backtesting work.

This repository now includes a minimal working scaffold so different issue owners can build on the same end-to-end pipeline instead of creating disconnected scripts.

## Folder Structure

```text
HighBuyLowSell/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── data_source_plan.md
├── notebooks/
├── outputs/
└── src/
    ├── __init__.py
    ├── check_raw_data.py
    ├── config.py
    ├── data.py
    ├── download_yahoo.py
    ├── features.py
    └── train_baseline.py
```

## Current Scope

- Primary data source: `Yahoo Finance / yfinance`
- Initial tickers: `NVDA`, `MSFT`, `GOOGL`, `BABA`, `BIDU`, `0700.HK`
- Start date: `2020-01-01`
- Raw schema: `date`, `open`, `high`, `low`, `close`, `volume`, `symbol`
- Main task: next-day direction classification
- Baseline models: Logistic Regression and XGBoost

## Quick Start

### 1) Install dependencies

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2) Download raw data

```bash
.venv/bin/python src/download_yahoo.py
```

### 3) Validate raw data

```bash
.venv/bin/python src/check_raw_data.py
```

### 4) Run a baseline model

```bash
.venv/bin/python src/train_baseline.py --input data/raw/yahoo_daily_prices.csv --symbol NVDA
```

Generated artifacts:

- `data/raw/yahoo_daily_prices.csv`
- `outputs/raw_data_summary.json`
- `data/processed/feature_table.csv`
- `outputs/metrics.json`
- `outputs/predictions.csv`

## Issue Mapping

- Issue 1: data source setup, schema validation, cleaning rules
- Issue 2: EDA and label sanity check
- Issue 3: feature engineering and baseline model comparison
- Issue 4: evaluation and simple backtesting

## Notes

- Keep raw data and generated outputs out of git.
- Respect time order in every split to avoid leakage.
- The scaffold is intentionally minimal. Feature expansion and evaluation logic should be added on top of this base rather than replacing it.
