# Data Source Plan

## Primary Source

- Source: `Yahoo Finance` via `yfinance`
- Purpose: download daily historical OHLCV data for the initial stock universe

## Initial Universe

Yahoo Finance ticker symbols used in the first-stage universe:

- `NVDA`
- `MSFT`
- `GOOGL`
- `BABA`
- `BIDU`
- `0700.HK`

## Locked Scope

- Start date: `2020-01-01`
- End date: latest trading day available from Yahoo Finance at execution time
- Interval: daily (`1d`)
- Stored columns:
  - `date`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`
  - `symbol`

## Field Definitions

- `symbol`: Yahoo Finance ticker used in the request
- `date`: trading date for the source market session, stored as a normalized daily timestamp without timezone information
- `open`, `high`, `low`, `close`: daily price fields returned by Yahoo Finance
- `volume`: daily trading volume returned by Yahoo Finance

## Price Adjustment Convention

- Data is downloaded through `yfinance`.
- If ingestion is configured with `auto_adjust=True`, then stored `open`, `high`, `low`, and `close` should be interpreted as split/dividend-adjusted prices as provided by Yahoo Finance.
- No separate `adj close` field is stored in the locked raw schema.

## Cross-Market Calendar Note

The initial universe spans multiple exchanges and market calendars, including U.S. and Hong Kong listings.

Therefore:

- trading dates will not always align across all symbols,
- a missing row for one symbol on a given calendar date is not automatically a data error,
- validation should be performed per symbol rather than assuming a perfectly shared trading calendar.

## Raw Layer Definition

The raw dataset is a lightly normalized extract from Yahoo Finance.   

At the raw layer:   

- requested symbols are attached explicitly as `symbol`,
- column names are normalized to the project schema,
- rows are sorted by `symbol` and `date`,
- unusable rows with missing essential OHLC fields may be removed during ingestion,
- no cross-symbol date alignment, forward filling, interpolation, or feature engineering is performed.

## Validation Rule      

After download, run the raw data check script to confirm:

1. requested tickers are present,
2. date ranges are sensible for each symbol,
3. duplicate (`symbol`, `date`) rows do not exist, and
4. missing values are visible before EDA starts.

## Validation Interpretation

Suggested interpretation of validation results:

### Error
- raw file cannot be read,
- dataset is empty,
- required columns are missing,
- duplicate (`symbol`, `date`) rows exist.

### Warning
- one or more expected tickers are missing,
- a symbol starts materially later than the requested start date,
- missing values appear in non-key fields such as `volume`,
- symbol coverage differs due to exchange-specific trading calendars.

## Natural Key

- The natural key of the raw daily price dataset is (`symbol`, `date`).
- This key is expected to be unique.

## Storage Target

- Raw download output: `data/raw/yahoo_daily_prices.csv`
- Raw validation summary: `outputs/raw_data_summary.json`

## Workflow

1. Download raw daily prices from Yahoo Finance.
2. Save the raw dataset to `data/raw/yahoo_daily_prices.csv`.
3. Run the raw data validation script.
4. Review validation warnings and errors before EDA or downstream processing.

## Reproducibility Note

- Because the end date is open-ended, repeated downloads at different times may produce different row counts and end dates.
- To improve reproducibility, record the download execution date and the exact ticker list used for each run.