import pandas as pd
import numpy as np

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build baseline features and a unified label for Issue 2.
    Strictly prevents look-ahead bias (data leakage). 
    All calculations must be performed within symbol groups.
    """
    # 1. Basic cleaning and sorting (Crucial: sort by ticker and time)
    # Assuming the input df contains: date, symbol, open, high, low, close, volume
    df = df.sort_values(by=['symbol', 'date']).copy()
    
    # 2. Core feature engineering (Use groupby to avoid cross-contamination)
    grouped = df.groupby('symbol')
    
    # - return_1d: 1-day percentage return
    df['return_1d'] = grouped['close'].pct_change(periods=1)
    
    # - return_5d: 5-day percentage return
    df['return_5d'] = grouped['close'].pct_change(periods=5)
    
    # - volume_change_1d: 1-day volume change rate
    df['volume_change_1d'] = grouped['volume'].pct_change(periods=1)
    
    # - ma_gap_5: Deviation from the 5-day moving average (bias ratio)
    # Calculate the MA from T-4 to T, then find the relative difference to the current price
    sma_5 = grouped['close'].transform(lambda x: x.rolling(window=5).mean())
    df['ma_gap_5'] = (df['close'] - sma_5) / sma_5
    
    # - ma_gap_10: Deviation from the 10-day moving average
    sma_10 = grouped['close'].transform(lambda x: x.rolling(window=10).mean())
    df['ma_gap_10'] = (df['close'] - sma_10) / sma_10
    
    # - volatility_5d: 5-day volatility (5-day rolling standard deviation of 1-day returns)
    df['volatility_5d'] = grouped['return_1d'].transform(lambda x: x.rolling(window=5).std())

    # 3. Label Definition
    # next-day direction classification: 1 for next day up, 0 for flat or down
    # Negative shift (shift(-1) to get tomorrow's data) is ONLY allowed here.
    future_close = grouped['close'].shift(-1)
    df['target_direction'] = (future_close > df['close']).astype(int)
    
    # Mark the natural NaNs created by shift(-1) at the end of each stock's series
    # Set them to NaN so they can be dropped together later
    df.loc[future_close.isna(), 'target_direction'] = np.nan

    # 4. Handle missing values
    # Rolling windows (e.g., 10-day MA) and shift operations create NaNs 
    # at the beginning and end of each stock's time series.
    df_clean = df.dropna().reset_index(drop=True)
    
    # Ensure target is integer type
    df_clean['target_direction'] = df_clean['target_direction'].astype(int)
    
    return df_clean

if __name__ == "__main__":
    # Local testing logic (will not interfere with downstream imports)
    try:
        raw_data = pd.read_csv("./data/raw/yahoo_daily_prices.csv")
        processed_data = build_baseline_features(raw_data)
        
        # Print baseline feature info for EDA reference
        print("=== Baseline Features Constructed ===")
        print(f"Original rows: {len(raw_data)} -> Cleaned rows: {len(processed_data)}")
        
        feature_cols = [col for col in processed_data.columns if col not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target_direction']]
        print(f"\nFeature list: {feature_cols}")
        print("\nLabel distribution (target_direction):")
        print(processed_data['target_direction'].value_counts(normalize=True))
        
        # Output for Issue 3 modeling team
        processed_data.to_csv("./data/processed/baseline_features.csv", index=False)
        print("\nSaved successfully to ./data/processed/baseline_features.csv")
        
    except FileNotFoundError:
        print("Raw data file not found. Please check the path.")