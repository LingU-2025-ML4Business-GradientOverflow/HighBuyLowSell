import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb

from feature_pipeline_universal import feature_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Traditional models training for stock prediction"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/yahoo_daily_prices.csv",
        help="Path to processed stock data",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=[
            "0700.HK",
            "BABA",
            "BIDU",
            "GOOGL",
            "MSFT",
            "NVDA",
        ],
        help="List of stock symbols to model",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ssm_uf_trad",
        help="Output directory for results",
    )
    return parser.parse_args()


def prepare_data(
    data_path: str, symbol: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for single stock modeling using feature pipeline

    Args:
        data_path: Path to data file
        symbol: Stock symbol
        test_size: Test set ratio
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load raw data
    full_data = pd.read_csv(data_path)

    # Filter specific stock data
    stock_data = full_data[full_data["symbol"] == symbol].copy()

    # Process data using feature pipeline
    processed_data = feature_pipeline.fit_transform(stock_data)

    # Define feature columns (exclude metadata and target)
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
    feature_cols = [c for c in processed_data.columns if c not in metadata_cols]

    # Prepare features and target
    X = processed_data[feature_cols]
    y = processed_data["target"]

    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> LogisticRegression:
    """Train logistic regression model"""
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> xgb.XGBClassifier:
    """Train XGBoost model with optimized parameters"""
    model = xgb.XGBClassifier(
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, symbol: str, model_name: str, output_dir: str) -> None:
    """
    Save trained model to disk

    Args:
        model: Trained model object
        symbol: Stock symbol
        model_name: Name of the model (logistic_regression or xgboost)
        output_dir: Output directory for models
    """
    import joblib

    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{symbol}_{model_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def run_single_stock_experiment(
    data_path: str,
    symbol: str,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "outputs/ssm_uf_trad",
) -> Dict:
    """
    Run single stock experiment using feature pipeline

    Args:
        data_path: Path to data file
        symbol: Stock symbol
        test_size: Test set ratio
        random_state: Random seed
        output_dir: Output directory for models

    Returns:
        Dictionary containing experiment results
    """
    print(f"Starting single stock experiment: {symbol}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        data_path, symbol, test_size, random_state
    )

    # Get feature column names
    feature_cols = X_train.columns.tolist()

    # Train models
    lr_model = train_logistic_regression(X_train, y_train, random_state)
    xgb_model = train_xgboost(X_train, y_train, random_state)

    # Save trained models
    save_model(lr_model, symbol, "logistic_regression", output_dir)
    save_model(xgb_model, symbol, "xgboost", output_dir)

    # Return results
    return {
        "symbol": symbol,
        "feature_cols": feature_cols,
    }


def main() -> None:
    """Main function"""
    args = parse_args()

    # Run experiments for all stocks
    all_results = []
    for symbol in args.symbols:
        result = run_single_stock_experiment(
            args.data_path, symbol, args.test_size, args.random_state, args.output_dir
        )
        all_results.append(result)

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Training results saved to {output_path}")


if __name__ == "__main__":
    main()
