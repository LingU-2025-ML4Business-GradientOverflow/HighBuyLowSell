# Data Source Plan

## Primary Source

- Source: `Yahoo Finance` via `yfinance`
- Purpose: daily historical OHLCV data for the initial stock universe

## Initial Universe

- `NVDA`
- `MSFT`
- `GOOGL`
- `BABA`
- `BIDU`
- `0700.HK`

## Locked Scope

- Start date: `2020-01-01`
- End date: latest available trading day
- Interval: daily (`1d`)
- Stored columns: `date`, `open`, `high`, `low`, `close`, `volume`, `symbol`

## Validation Rule

After download, run the raw data check script to confirm:

1. requested tickers are present,
2. date ranges are sensible for each symbol,
3. duplicate `(symbol, date)` rows do not exist, and
4. missing values are visible before EDA starts.
