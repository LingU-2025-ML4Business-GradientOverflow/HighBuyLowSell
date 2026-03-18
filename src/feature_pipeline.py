import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def safe_divide(numerator, denominator, fill_value=0.0, epsilon=1e-8):
    """Safely divide two arrays/series, handling zero denominators."""
    result = np.where(
        np.abs(denominator) < epsilon,
        fill_value,
        numerator / (denominator + epsilon)
    )
    return result

class MacroeconomicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Step 1: Add macroeconomic features.
    Fetches S&P500, NASDAQ, Interest Rate (10Y), and VIX indices.
    """
    def __init__(self, auto_download=True):
        self.auto_download = auto_download
        self.market_data = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        if self.auto_download:
            print("Downloading macroeconomic features (S&P500, NASDAQ, Interest Rate, VIX)...")
            try:
                self.market_data = self._fetch_market_data(
                    df['date'].min(), 
                    df['date'].max()
                )
            except Exception as e:
                print(f"Warning: Failed to download market data: {e}")
                self._create_empty_features(df)
                return df
        
        # Merge market data with stock data
        df = df.merge(self.market_data, on='date', how='left')
        
        # Forward fill for missing values (weekends/holidays) - Pandas 2.0+ compatible
        for col in ['sp500', 'nasdaq', 'interest_rate', 'vix']:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        return df
    
    def _fetch_market_data(self, start_date, end_date):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        indices = {
            'sp500': '^GSPC', 'nasdaq': '^IXIC', 'interest_rate': '^TNX', 'vix': '^VIX'
        }
        
        market_data = None
        for feature_name, ticker in indices.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data.empty: continue
                
                df_temp = data[['Close']].reset_index()
                df_temp.columns = ['date', feature_name]
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                
                if market_data is None:
                    market_data = df_temp
                else:
                    market_data = market_data.merge(df_temp, on='date', how='outer')
            except Exception:
                continue
        
        if market_data is None:
            raise ValueError("Failed to download any market data")
        
        return market_data.sort_values('date').reset_index(drop=True)

    def _create_empty_features(self, df):
        for col in ['sp500', 'nasdaq', 'interest_rate', 'vix']:
            df[col] = 0.0

class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Step 2: Calculate basic technical indicators."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        
        grouped = df.groupby('symbol')
        df['return_1d'] = grouped['close'].pct_change()
        df['return_5d'] = grouped['close'].pct_change(5)
        df['volume_change_1d'] = grouped['volume'].pct_change()
        
        df['ma_5'] = grouped['close'].transform(lambda x: x.rolling(5).mean())
        df['ma_10'] = grouped['close'].transform(lambda x: x.rolling(10).mean())
        df['ma_gap_5'] = safe_divide(df['close'] - df['ma_5'], df['ma_5'])
        df['ma_gap_10'] = safe_divide(df['close'] - df['ma_10'], df['ma_10'])
        df['volatility_5d'] = grouped['return_1d'].transform(lambda x: x.rolling(5).std())

        return df.dropna().reset_index(drop=True)

class AdvancedCompanySpecificTransformer(BaseEstimator, TransformerMixin):
    """Step 3: Generate 18 specific features tailored to each company."""
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
        for col in specific_cols: df[col] = 0.0

        # FIXED LOGIC: Applied masks (.loc) inside safe_divide and .rolling() to prevent shape mismatch
        nvda_m = df['symbol'] == 'NVDA'
        if nvda_m.any():
            df.loc[nvda_m, 'nvda_compute_frenzy'] = df.loc[nvda_m, 'volume'] * df.loc[nvda_m, 'return_1d'].abs()
            df.loc[nvda_m, 'nvda_supply_momentum'] = safe_divide(df.loc[nvda_m, 'return_5d'], df.loc[nvda_m, 'volatility_5d'])
            df.loc[nvda_m, 'nvda_price_overheat'] = safe_divide(df.loc[nvda_m, 'close'], df.loc[nvda_m, 'close'].rolling(20).mean())

        msft_m = df['symbol'] == 'MSFT'
        if msft_m.any():
            df.loc[msft_m, 'msft_efficiency_ratio'] = safe_divide(df.loc[msft_m, 'return_5d'], df.loc[msft_m, 'return_1d'].abs().rolling(5).sum())
            df.loc[msft_m, 'msft_trend_consistency'] = df.loc[msft_m, 'ma_gap_10'] - df.loc[msft_m, 'ma_gap_5']
            df.loc[msft_m, 'msft_inst_accumulation'] = df.loc[msft_m, 'volume'] * df.loc[msft_m, 'close']

        googl_m = df['symbol'] == 'GOOGL'
        if googl_m.any():
            df.loc[googl_m, 'googl_ad_volatility'] = df.loc[googl_m, 'volatility_5d'].rolling(10).std()
            df.loc[googl_m, 'googl_mean_reversion'] = safe_divide(df.loc[googl_m, 'close'] - df.loc[googl_m, 'close'].rolling(20).mean(), df.loc[googl_m, 'close'].rolling(20).std())
            df.loc[googl_m, 'googl_liquidity_density'] = safe_divide(df.loc[googl_m, 'volume'], df.loc[googl_m, 'high'] - df.loc[googl_m, 'low'])

        hk_m = df['symbol'] == '0700.HK'
        if hk_m.any():
            df.loc[hk_m, 'tencent_policy_shock'] = df.loc[hk_m, 'volume_change_1d'].abs() * df.loc[hk_m, 'return_1d']
            df.loc[hk_m, 'tencent_capital_inflow'] = safe_divide(df.loc[hk_m, 'volume'], df.loc[hk_m, 'volume'].rolling(20).mean())
            df.loc[hk_m, 'tencent_gap_dynamic'] = safe_divide(df.loc[hk_m, 'open'] - df.loc[hk_m, 'close'].shift(1), df.loc[hk_m, 'close'].shift(1))

        baba_m = df['symbol'] == 'BABA'
        if baba_m.any():
            df.loc[baba_m, 'baba_retail_inflection'] = df.loc[baba_m, 'ma_gap_5'].diff()
            df.loc[baba_m, 'baba_vol_clustering'] = safe_divide(df.loc[baba_m, 'volatility_5d'], df.loc[baba_m, 'volatility_5d'].rolling(20).mean())
            df.loc[baba_m, 'baba_pv_divergence'] = df.loc[baba_m, 'volume_change_1d'] - df.loc[baba_m, 'return_1d']

        bidu_m = df['symbol'] == 'BIDU'
        if bidu_m.any():
            df.loc[bidu_m, 'bidu_ai_breakout'] = df.loc[bidu_m, 'return_1d'] * df.loc[bidu_m, 'volume_change_1d']
            df.loc[bidu_m, 'bidu_cash_stability'] = safe_divide(df.loc[bidu_m, 'close'].rolling(10).median(), df.loc[bidu_m, 'close'])
            df.loc[bidu_m, 'bidu_rebound_force'] = safe_divide(df.loc[bidu_m, 'low'].rolling(20).min(), df.loc[bidu_m, 'close'])

        return df.fillna(0)

# Complete Pipeline
feature_pipeline = Pipeline([
    ('macro_features', MacroeconomicFeaturesTransformer(auto_download=True)),
    ('basic_indicators', TechnicalIndicatorTransformer()),
    ('advanced_specifics', AdvancedCompanySpecificTransformer())
])

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    
    input_file = project_root / "data" / "raw" / "yahoo_daily_prices.csv"
    output_dir = project_root / "data" / "processed"
    
    if not input_file.exists():
        print(f"CRITICAL ERROR: Input file not found at {input_file}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_data = pd.read_csv(input_file)
            processed_df = feature_pipeline.fit_transform(raw_data)
            output_path = output_dir / "advanced_features_output.csv"
            processed_df.to_csv(output_path, index=False)
            print(f"Success! Final Shape: {processed_df.shape}")
            print(f"File saved to: {output_path}")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()