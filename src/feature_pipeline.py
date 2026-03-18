import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Step 1: Calculate basic technical indicators.
    Ensures data consistency and handles general market trends.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure data is sorted for time-series calculations
        df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        
        # We assume baseline_features.csv already has return_1d, volatility_5d, etc.
        # If any calculation is missing, it can be added here using grouped.transform()
        return df

class AdvancedCompanySpecificTransformer(BaseEstimator, TransformerMixin):
    """
    Step 2: Synthesize 3 exclusive dimensions per company (18 total columns).
    Uses a sparse matrix approach: fills non-target company rows with 0.0.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # --- Pre-initialize all specific columns with 0.0 ---
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

        # --- 1. NVDA (AI Infrastructure & Compute) ---
        nvda_mask = df['symbol'] == 'NVDA'
        # Frenzy: Volume multiplied by absolute daily return
        df.loc[nvda_mask, 'nvda_compute_frenzy'] = df['volume'] * df['return_1d'].abs()
        # Supply: Risk-adjusted return over 5 days
        df.loc[nvda_mask, 'nvda_supply_momentum'] = df['return_5d'] / (df['volatility_5d'] + 1e-6)
        # Overheat: Distance from the 20-day mean
        df.loc[nvda_mask, 'nvda_price_overheat'] = df['close'] / df['close'].rolling(20).mean()

        # --- 2. MSFT (B2B & Enterprise Stability) ---
        msft_mask = df['symbol'] == 'MSFT'
        # Efficiency: Path efficiency of price movement (Net move vs total volatility)
        df.loc[msft_mask, 'msft_efficiency_ratio'] = df['return_5d'] / (df['return_1d'].abs().rolling(5).sum() + 1e-6)
        # Trend: Convergence/Divergence of short and mid-term momentum
        df.loc[msft_mask, 'msft_trend_consistency'] = df['ma_gap_10'] - df['ma_gap_5']
        # Accumulation: Proxy for institutional dollar volume
        df.loc[msft_mask, 'msft_inst_accumulation'] = df['volume'] * df['close']

        # --- 3. GOOGL (Ad Market & Search Volume) ---
        googl_mask = df['symbol'] == 'GOOGL'
        # Ad Volatility: Standard deviation of short-term volatility
        df.loc[googl_mask, 'googl_ad_volatility'] = df['volatility_5d'].rolling(10).std()
        # Mean Reversion: Z-score relative to the 20-day moving average
        df.loc[googl_mask, 'googl_mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-6)
        # Liquidity: Volume per unit of price range
        df.loc[googl_mask, 'googl_liquidity_density'] = df['volume'] / (df['high'] - df['low'] + 1e-6)

        # --- 4. 0700.HK (Policy & HK Market Liquidity) ---
        hk0700_mask = df['symbol'] == '0700.HK'
        # Policy: Captures return direction during volume surges
        df.loc[hk0700_mask, 'tencent_policy_shock'] = df['volume_change_1d'].abs() * df['return_1d']
        # Capital Inflow: Current volume relative to 20-day average volume
        df.loc[hk0700_mask, 'tencent_capital_inflow'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)
        # Gap Dynamic: Overnight gap relative to previous close
        df.loc[hk0700_mask, 'tencent_gap_dynamic'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-6)

        # --- 5. BABA (Consumer Discretionary & Retail Cycle) ---
        baba_mask = df['symbol'] == 'BABA'
        # Inflection: Rate of change of the MA Gap
        df.loc[baba_mask, 'baba_retail_inflection'] = df['ma_gap_5'].diff()
        # Clustering: Current volatility relative to monthly average
        df.loc[baba_mask, 'baba_vol_clustering'] = df['volatility_5d'] / (df['volatility_5d'].rolling(20).mean() + 1e-6)
        # Divergence: Spread between volume change and price change
        df.loc[baba_mask, 'baba_pv_divergence'] = df['volume_change_1d'] - df['return_1d']

        # --- 6. BIDU (AI Pivot & Technology Transformation) ---
        bidu_mask = df['symbol'] == 'BIDU'
        # Breakout: Price and volume synchronization
        df.loc[bidu_mask, 'bidu_ai_breakout'] = df['return_1d'] * df['volume_change_1d']
        # Stability: Median price support vs current price
        df.loc[bidu_mask, 'bidu_cash_stability'] = df['close'].rolling(10).median() / df['close']
        # Rebound: Distance from the 20-day low
        df.loc[bidu_mask, 'bidu_rebound_force'] = df['low'].rolling(20).min() / df['close']

        return df.fillna(0)

# Instantiate the Pipeline
feature_pipeline = Pipeline([
    ('indicators', TechnicalIndicatorTransformer()),
    ('advanced_specifics', AdvancedCompanySpecificTransformer())
])

if __name__ == "__main__":
    try:
        # 1. Load your baseline data
        print("Loading baseline_features.csv...")
        raw_data = pd.read_csv("baseline_features.csv")
        
        # 2. Run Pipeline
        print("Processing features for 6 companies (18 specific dimensions)...")
        processed_df = feature_pipeline.fit_transform(raw_data)
        
        # 3. Output results and basic verification
        print("\n=== Pipeline Execution Summary ===")
        print(f"Final Data Shape: {processed_df.shape}")
        
        # Verification: Check if NVDA features are isolated from MSFT
        nvda_val = processed_df.loc[processed_df['symbol'] == 'NVDA', 'nvda_compute_frenzy'].mean()
        msft_val = processed_df.loc[processed_df['symbol'] == 'MSFT', 'nvda_compute_frenzy'].mean()
        
        print(f"Mean 'nvda_compute_frenzy' for NVDA: {nvda_val:.4f}")
        print(f"Mean 'nvda_compute_frenzy' for MSFT (Should be 0.0): {msft_val:.4f}")
        
        # 4. Save to CSV
        output_file = "advanced_features_output.csv"
        processed_df.to_csv(output_file, index=False)
        print(f"\nSuccess! Processed data saved to '{output_file}'")
        
    except FileNotFoundError:
        print("Error: baseline_features.csv not found in the current directory.")