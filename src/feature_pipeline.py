import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Step 1: Calculate basic technical indicators.
    Ensures that the basic columns exist for the specific logic to follow.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Convert date to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        
        grouped = df.groupby('symbol')

        # Basic Indicators
        df['return_1d'] = grouped['close'].pct_change()
        df['return_5d'] = grouped['close'].pct_change(5)
        df['volume_change_1d'] = grouped['volume'].pct_change()
        
        df['ma_5'] = grouped['close'].transform(lambda x: x.rolling(window=5).mean())
        df['ma_10'] = grouped['close'].transform(lambda x: x.rolling(window=10).mean())
        df['ma_gap_5'] = (df['close'] - df['ma_5']) / (df['ma_5'] + 1e-6)
        df['ma_gap_10'] = (df['close'] - df['ma_10']) / (df['ma_10'] + 1e-6)
        
        df['volatility_5d'] = grouped['return_1d'].transform(lambda x: x.rolling(window=5).std())

        # Important: drop first few rows where indicators are NaN
        return df.dropna().reset_index(drop=True)


class AdvancedCompanySpecificTransformer(BaseEstimator, TransformerMixin):
    """
    Step 2: Generate 3 specific features per company (18 total).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        specific_cols = [
            'nvda_compute_frenzy', 'nvda_supply_momentum', 'nvda_price_overheat',
            'msft_efficiency_ratio', 'msft_trend_consistency', 'msft_inst_accumulation',
            'googl_ad_volatility', 'googl_mean_reversion', 'googl_liquidity_density',
            'tencent_policy_shock', 'tencent_capital_inflow', 'tencent_gap_dynamic',
            'baba_retail_inflection', 'baba_vol_clustering', 'baba_pv_divergence',
            'bidu_ai_breakout', 'bidu_cash_stability', 'bidu_rebound_force'
        ]
        for col in specific_cols:
            df[col] = 0.0

        # 1. NVDA
        nvda_m = df['symbol'] == 'NVDA'
        df.loc[nvda_m, 'nvda_compute_frenzy'] = df['volume'] * df['return_1d'].abs()
        df.loc[nvda_m, 'nvda_supply_momentum'] = df['return_5d'] / (df['volatility_5d'] + 1e-6)
        df.loc[nvda_m, 'nvda_price_overheat'] = df['close'] / (df['close'].rolling(20).mean() + 1e-6)

        # 2. MSFT
        msft_m = df['symbol'] == 'MSFT'
        df.loc[msft_m, 'msft_efficiency_ratio'] = df['return_5d'] / (df['return_1d'].abs().rolling(5).sum() + 1e-6)
        df.loc[msft_m, 'msft_trend_consistency'] = df['ma_gap_10'] - df['ma_gap_5']
        df.loc[msft_m, 'msft_inst_accumulation'] = df['volume'] * df['close']

        # 3. GOOGL
        googl_m = df['symbol'] == 'GOOGL'
        df.loc[googl_m, 'googl_ad_volatility'] = df['volatility_5d'].rolling(10).std()
        df.loc[googl_m, 'googl_mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-6)
        df.loc[googl_m, 'googl_liquidity_density'] = df['volume'] / (df['high'] - df['low'] + 1e-6)

        # 4. 0700.HK (Tencent)
        hk_m = df['symbol'] == '0700.HK'
        df.loc[hk_m, 'tencent_policy_shock'] = df['volume_change_1d'].abs() * df['return_1d']
        df.loc[hk_m, 'tencent_capital_inflow'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)
        df.loc[hk_m, 'tencent_gap_dynamic'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-6)

        # 5. BABA
        baba_m = df['symbol'] == 'BABA'
        df.loc[baba_m, 'baba_retail_inflection'] = df['ma_gap_5'].diff()
        df.loc[baba_m, 'baba_vol_clustering'] = df['volatility_5d'] / (df['volatility_5d'].rolling(20).mean() + 1e-6)
        df.loc[baba_m, 'baba_pv_divergence'] = df['volume_change_1d'] - df['return_1d']

        # 6. BIDU
        bidu_m = df['symbol'] == 'BIDU'
        df.loc[bidu_m, 'bidu_ai_breakout'] = df['return_1d'] * df['volume_change_1d']
        df.loc[bidu_m, 'bidu_cash_stability'] = df['close'].rolling(10).median() / (df['close'] + 1e-6)
        df.loc[bidu_m, 'bidu_rebound_force'] = df['low'].rolling(20).min() / (df['close'] + 1e-6)


        # --- Target Definition (Next day direction) ---
        # Next-day close > Today's close
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        return df.fillna(0)


# Instantiate Pipeline
feature_pipeline = Pipeline([
    ('basic_indicators', TechnicalIndicatorTransformer()),
    ('advanced_specifics', AdvancedCompanySpecificTransformer())
])


if __name__ == "__main__":
    # --- PATH LOGIC (Cross-platform compatible) ---
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Navigate to project root (assuming src/ is inside project root)
    project_root = script_dir.parent
    
    # Define input and output paths
    input_file = project_root / "data" / "raw" / "yahoo_daily_prices.csv"
    output_dir = project_root / "data" / "processed"
    
    # Validate input file exists
    if not input_file.exists():
        print(f"CRITICAL ERROR: Input file not found at {input_file}")
        print("Please ensure the data file exists in data/raw/")
    else:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Loading data from: {input_file}")
            raw_data = pd.read_csv(input_file)
            
            print("Running feature engineering pipeline...")
            processed_df = feature_pipeline.fit_transform(raw_data)
            
            output_path = output_dir / "advanced_features_output.csv"
            processed_df.to_csv(output_path, index=False)
            
            print(f"\nSuccess! Final Shape: {processed_df.shape}")
            print(f"File saved to: {output_path}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()