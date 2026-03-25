import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from feature_pipeline_universal import feature_pipeline


# Define evaluation metrics
METRICS = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Single stock modeling exploration")
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


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的概率
    results = {}

    for metric_name, metric_func in METRICS.items():
        try:
            if metric_name == "roc_auc":
                results[metric_name] = metric_func(y_test, y_pred_prob)
            else:
                results[metric_name] = metric_func(y_test, y_pred)
        except Exception as e:
            print(f"Error calculating {metric_name}: {str(e)}")
            results[metric_name] = 0.0

    return results


def get_feature_columns(data_path: str) -> List[str]:
    """
    Get feature column names from feature pipeline

    Args:
        data_path: Path to data file

    Returns:
        List of feature column names
    """
    # Use sample data to get feature columns
    sample_data = pd.read_csv(data_path).head(100)
    processed_sample = feature_pipeline.fit_transform(sample_data)

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
    return [c for c in processed_sample.columns if c not in metadata_cols]


def run_single_stock_experiment(
    data_path: str, symbol: str, test_size: float = 0.2, random_state: int = 42
) -> Dict:
    """
    Run single stock experiment using feature pipeline

    Args:
        data_path: Path to data file
        symbol: Stock symbol
        test_size: Test set ratio
        random_state: Random seed

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

    # Evaluate models
    lr_results = evaluate_model(lr_model, X_test, y_test)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)

    # Return results
    return {
        "symbol": symbol,
        "logistic_regression": lr_results,
        "xgboost": xgb_results,
        "feature_importance": {
            "logistic_regression": dict(zip(feature_cols, lr_model.coef_[0])),
            "xgboost": dict(zip(feature_cols, xgb_model.feature_importances_)),
        },
    }


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """Compare results across different stocks and models"""
    comparison_data = []

    for result in results:
        symbol = result["symbol"]
        for model_name, metrics in result.items():
            if model_name in ["logistic_regression", "xgboost"]:
                row = {"symbol": symbol, "model": model_name}
                row |= metrics
                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Print key information to console
    print("\n" + "=" * 50)
    print("compare_models")
    print("=" * 50)
    print("\nbest combination:")
    best = comparison_df.loc[comparison_df["roc_auc"].idxmax()]
    print(
        f"  Stock: {best['symbol']}, Model: {best['model']}, ROC-AUC: {best['roc_auc']:.4f}"
    )

    print("\nAverage ROC-AUC by Model:")
    for model in comparison_df["model"].unique():
        avg_roc = comparison_df[comparison_df["model"] == model]["roc_auc"].mean()
        print(f"  {model}: {avg_roc:.4f}")

    return comparison_df


def save_results(results: List[Dict], output_dir: str) -> None:
    """Save experiment results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison_df = compare_models(results)
    comparison_df.to_csv(output_path / "model_comparison.csv", index=False)

    # Save detailed results for each stock
    for result in results:
        symbol = result["symbol"]
        symbol_path = output_path / f"{symbol}_results.json"

        # Convert DataFrame to dict for JSON serialization
        import json

        with open(symbol_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    print(f"Results saved to {output_path}")


def generate_conclusions(results: List[Dict]) -> Dict:
    """Generate experiment conclusions"""
    comparison_df = compare_models(results)

    # Find best performing stock and model based on ROC-AUC
    best_roc = comparison_df.loc[comparison_df["roc_auc"].idxmax()]

    # Compare model stability using ROC-AUC
    model_stability = comparison_df.groupby("model")["roc_auc"].std().to_dict()

    # Compare stock modelability using ROC-AUC
    stock_modelability = comparison_df.groupby("symbol")["roc_auc"].mean().to_dict()

    return {
        "best_performing": {
            "symbol": best_roc["symbol"],
            "model": best_roc["model"],
            "roc_auc": best_roc["roc_auc"],
        },
        "model_stability": model_stability,
        "stock_modelability": stock_modelability,
        "recommendations": {
            "worth_retaining": model_stability["xgboost"]
            < model_stability["logistic_regression"],
            "recommended_symbol": max(stock_modelability, key=stock_modelability.get),
            "recommended_model": (
                "xgboost"
                if model_stability["xgboost"] < model_stability["logistic_regression"]
                else "logistic_regression"
            ),
        },
    }


def main() -> None:
    """Main function"""
    args = parse_args()

    # Run experiments for all stocks
    all_results = []
    for symbol in args.symbols:
        result = run_single_stock_experiment(
            args.data_path, symbol, args.test_size, args.random_state
        )
        all_results.append(result)

    # Save results
    save_results(all_results, args.output_dir)

    # Generate conclusions
    conclusions = generate_conclusions(all_results)

    # Print conclusions
    print("Experiment Conclusions:")
    print(f"Best Performance: {conclusions['best_performing']}")
    print(f"Model Stability: {conclusions['model_stability']}")
    print(f"Stock Modelability: {conclusions['stock_modelability']}")
    print(f"Recommendations: {conclusions['recommendations']}")


if __name__ == "__main__":
    main()
