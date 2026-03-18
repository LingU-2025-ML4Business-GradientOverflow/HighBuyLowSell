import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def safe_divide(numerator, denominator, fill_value=0.0, epsilon=1e-8):
    """
    Safely divide two arrays/series, handling zero denominators.
    
    Parameters:
    - numerator: numerator values
    - denominator: denominator values
    - fill_value: replacement value when denominator is zero (default 0.0)
    - epsilon: small value added to denominator to prevent numerical instability
    """
    result = np.where(
        np.abs(denominator) < epsilon,
        fill_value,
        numerator / (denominator + epsilon)
    )
    return result


class MacroeconomicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Step 1a: Add macroeconomic features.
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
                print("Using zero values for macroeconomic features")
                self._create_empty_features(df)
                return df
        
        # Merge market data with stock data
        df = df.merge(self.market_data, on='date', how='left')
        
        # Forward fill for missing values (weekends/holidays)
        for col in ['sp500', 'nasdaq', 'interest_rate', 'vix']:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        return df
    
    def _fetch_market_data(self, start_date, end_date):
        """Download market indices from Yahoo Finance"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
        
        indices = {
            'sp500': '^GSPC',      # S&P 500
            'nasdaq': '^IXIC',     # NASDAQ
            'interest_rate': '^TNX', # 10-Year Treasury Yield
            'vix': '^VIX'          # Volatility Index
        }
        
        market_data = None
        
        for feature_name, ticker in indices.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data.empty:
                    continue
                
                df_temp = data[['Close']].reset_index()
                df_temp.columns = ['date', feature_name]
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                
                if market_data is None:
                    market_data = df_temp
                else:
                    market_data = market_data.merge(df_temp, on='date', how='outer')
                    
            except Exception as e:
                print(f"Warning: Failed to download {feature_name} ({ticker}): {e}")
        
        if market_data is None:
            raise ValueError("Failed to download any market data")
        
        market_data = market_data.sort_values('date').reset_index(drop=True)
        return market_data
    
    def _create_empty_features(self, df):
        """Create zero-valued features as fallback when download fails"""
        for col in ['sp500', 'nasdaq', 'interest_rate', 'vix']:
            df[col] = 0.0


class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Step 2: Calculate basic technical indicators.
    Computes returns, moving averages, volatility, and related metrics.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Convert date column to datetime and sort by symbol and date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['symbol', 'date']).reset_index(drop=True)
        
        grouped = df.groupby('symbol')

        # Compute basic technical indicators
        df['return_1d'] = grouped['close'].pct_change()
        df['return_5d'] = grouped['close'].pct_change(5)
        df['volume_change_1d'] = grouped['volume'].pct_change()
        
        df['ma_5'] = grouped['close'].transform(lambda x: x.rolling(window=5).mean())
        df['ma_10'] = grouped['close'].transform(lambda x: x.rolling(window=10).mean())
        df['ma_gap_5'] = safe_divide(df['close'] - df['ma_5'], df['ma_5'])
        df['ma_gap_10'] = safe_divide(df['close'] - df['ma_10'], df['ma_10'])
        
        df['volatility_5d'] = grouped['return_1d'].transform(lambda x: x.rolling(window=5).std())

        # Drop rows with NaN values (typically first few rows due to rolling windows)
        return df.dropna().reset_index(drop=True)


class AdvancedCompanySpecificTransformer(BaseEstimator, TransformerMixin):
    """
    Step 3: Generate company-specific features (3 per company, 18 total).
    Creates customized indicators tailored to each stock's characteristics.
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

        # 1. NVDA - Graphics Processing & AI Hardware
        nvda_m = df['symbol'] == 'NVDA'
        df.loc[nvda_m, 'nvda_compute_frenzy'] = df['volume'] * df['return_1d'].abs()
        df.loc[nvda_m, 'nvda_supply_momentum'] = safe_divide(df['return_5d'], df['volatility_5d'])
        df.loc[nvda_m, 'nvda_price_overheat'] = safe_divide(df['close'], df['close'].rolling(20).mean())

        # 2. MSFT - Cloud Computing & Enterprise Software
        msft_m = df['symbol'] == 'MSFT'
        df.loc[msft_m, 'msft_efficiency_ratio'] = safe_divide(df['return_5d'], df['return_1d'].abs().rolling(5).sum())
        df.loc[msft_m, 'msft_trend_consistency'] = df['ma_gap_10'] - df['ma_gap_5']
        df.loc[msft_m, 'msft_inst_accumulation'] = df['volume'] * df['close']

        # 3. GOOGL - Digital Advertising & Cloud Services
        googl_m = df['symbol'] == 'GOOGL'
        df.loc[googl_m, 'googl_ad_volatility'] = df['volatility_5d'].rolling(10).std()
        df.loc[googl_m, 'googl_mean_reversion'] = safe_divide(
            df['close'] - df['close'].rolling(20).mean(), 
            df['close'].rolling(20).std()
        )
        df.loc[googl_m, 'googl_liquidity_density'] = safe_divide(df['volume'], df['high'] - df['low'])

        # 4. 0700.HK (Tencent) - Gaming, Social Media & Cloud
        hk_m = df['symbol'] == '0700.HK'
        df.loc[hk_m, 'tencent_policy_shock'] = df['volume_change_1d'].abs() * df['return_1d']
        df.loc[hk_m, 'tencent_capital_inflow'] = safe_divide(df['volume'], df['volume'].rolling(20).mean())
        df.loc[hk_m, 'tencent_gap_dynamic'] = safe_divide(df['open'] - df['close'].shift(1), df['close'].shift(1))

        # 5. BABA - E-commerce & Digital Economy
        baba_m = df['symbol'] == 'BABA'
        df.loc[baba_m, 'baba_retail_inflection'] = df['ma_gap_5'].diff()
        df.loc[baba_m, 'baba_vol_clustering'] = safe_divide(df['volatility_5d'], df['volatility_5d'].rolling(20).mean())
        df.loc[baba_m, 'baba_pv_divergence'] = df['volume_change_1d'] - df['return_1d']

        # 6. BIDU - Search Engine & AI Technology
        bidu_m = df['symbol'] == 'BIDU'
        df.loc[bidu_m, 'bidu_ai_breakout'] = df['return_1d'] * df['volume_change_1d']
        df.loc[bidu_m, 'bidu_cash_stability'] = safe_divide(df['close'].rolling(10).median(), df['close'])
        df.loc[bidu_m, 'bidu_rebound_force'] = safe_divide(df['low'].rolling(20).min(), df['close'])

        return df.fillna(0)


# Instantiate the complete feature engineering pipeline
feature_pipeline = Pipeline([
    ('macro_features', MacroeconomicFeaturesTransformer(auto_download=True)),
    ('basic_indicators', TechnicalIndicatorTransformer()),
    ('advanced_specifics', AdvancedCompanySpecificTransformer())
])


if __name__ == "__main__":
    # --- PATH LOGIC (Cross-platform compatible) ---
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Navigate to project root (assumes src/ is inside project root)
    project_root = script_dir.parent
    
    # Define input and output file paths
    input_file = project_root / "data" / "raw" / "yahoo_daily_prices.csv"
    output_dir = project_root / "data" / "processed"
    
    # Validate that input file exists
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
            
            print(f"\nSuccess! Final dataset shape: {processed_df.shape}")
            print(f"Output file saved to: {output_path}")
            
        except Exception as e:
            print(f"An error occurred during pipeline execution: {e}")
            import traceback
            traceback.print_exc()