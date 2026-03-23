import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TechnicalIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate all required technical indicators for Issue 2.
    Ensures no look-ahead bias by using group-based windowing functions.

    """

    def __init__(self):
        # No hyperparameters needed for this baseline
        pass

    def fit(self, X, y=None):
        """
        In scikit-learn, fit must return self.
        Even if no parameters are learned, this 'activates' the transformer.
        """
        return self

    def transform(self, X):
        """
        Core logic to calculate 10+ technical indicators.
        Input: Raw OHLCV DataFrame
        Output: Processed DataFrame with features and target
        """
        df = X.copy()
        # Ensure data is sorted for time-series calculations
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        grouped = df.groupby("symbol")

        # --- 1. Trend Indicators ---
        # SMA 5, 10, 20
        for n in [5, 10, 20]:
            df[f"sma_{n}"] = grouped["close"].transform(
                lambda x: x.rolling(window=n).mean()
            )

        # EMA 12, 26
        df["ema_12"] = grouped["close"].transform(
            lambda x: x.ewm(span=12, adjust=False).mean()
        )
        df["ema_26"] = grouped["close"].transform(
            lambda x: x.ewm(span=26, adjust=False).mean()
        )

        # MACD (Moving Average Convergence Divergence)
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = grouped["macd"].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- 2. Momentum Indicators ---
        # RSI 14 (Relative Strength Index)
        def calc_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-9)  # Avoid division by zero
            return 100 - (100 / (1 + rs))

        df["rsi_14"] = grouped["close"].transform(calc_rsi)

        # --- 3. Volatility Indicators ---
        # Bollinger Bands (20, 2)
        df["bb_mid"] = df["sma_20"]
        df["bb_std"] = grouped["close"].transform(lambda x: x.rolling(window=20).std())
        df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * 2)

        # --- 4. Volume Indicators ---
        # Simple Price-Volume Divergence: Price Up AND Volume Down
        df["price_up"] = (df["close"] > grouped["close"].shift(1)).astype(int)
        df["vol_down"] = (df["volume"] < grouped["volume"].shift(1)).astype(int)
        df["vol_price_divergence"] = (df["price_up"] & df["vol_down"]).astype(int)

        # --- 5. Return & Volatility Features (Baseline) ---
        df["return_1d"] = grouped["close"].pct_change(1)
        df["ma_gap_10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
        df["volatility_5d"] = grouped["return_1d"].transform(
            lambda x: x.rolling(window=5).std()
        )

        # --- 6. Target Definition (Next day direction) ---
        # Next-day close > Today's close
        df["target"] = (grouped["close"].shift(-1) > df["close"]).astype(int)

        # Handle end-of-series NaNs for target and start-of-series NaNs for indicators
        # SMA20/RSI14/BB will create about 20 NaNs at the start of each symbol
        df_clean = df.dropna().reset_index(drop=True)
        return df_clean


# Instantiate the Pipeline
# This is what Issue 3A/3B will import
feature_pipeline = Pipeline([("indicators", TechnicalIndicatorTransformer())])

if __name__ == "__main__":
    # Test script for Issue 2 verification
    try:
        # 1. Load data
        raw_data = pd.read_csv("./data/raw/yahoo_daily_prices.csv")

        # 2. RUN PIPELINE: Use fit_transform instead of transform
        print("Fitting and transforming data through pipeline...")
        processed_df = feature_pipeline.fit_transform(raw_data)

        # 3. Output results
        print("\n=== Pipeline Output Statistics ===")
        print(f"Final Data Shape: {processed_df.shape}")

        # List generated features (excluding standard columns and target)
        metadata_cols = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
        ]
        features_list = [c for c in processed_df.columns if c not in metadata_cols]
        print(f"Features Generated ({len(features_list)}): {features_list}")

        # 4. Export for modeling team
        processed_df.to_csv("./data/processed/features.csv", index=False)
        print("\nSUCCESS: features.csv has been updated.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
