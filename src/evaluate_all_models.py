import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split

from feature_pipeline import feature_pipeline
from feature_pipeline_universal import feature_pipeline as feature_pipeline_universal
from StockCNN import StockCNN

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
    parser = argparse.ArgumentParser(
        description="All models evaluation for stock prediction"
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
        help="List of stock symbols to evaluate",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs",
        help="Directory containing saved models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/all_models",
        help="Output directory for results",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def prepare_data(
    data_path: str,
    symbol: str,
    pipeline_type: str = "universal",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for single stock modeling using feature pipeline

    Args:
        data_path: Path to data file
        symbol: Stock symbol
        pipeline_type: Type of feature pipeline to use ("universal" or "specific")
        test_size: Test set ratio
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Load raw data
    full_data = pd.read_csv(data_path)

    # Filter specific stock data
    stock_data = full_data[full_data["symbol"] == symbol].copy()

    # Process data using appropriate feature pipeline
    if pipeline_type == "universal":
        processed_data = feature_pipeline_universal.fit_transform(stock_data)
    else:
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

    # Handle infinite values for CNN models
    if pipeline_type == "universal":
        X_train.replace((np.inf, -np.inf), 0, inplace=True)

    return X_train, X_test, y_train, y_test


def load_traditional_model(model_path: str):
    """Load a saved traditional model from disk"""
    model = joblib.load(model_path)
    print(f"Traditional model loaded from {model_path}")
    return model


def load_cnn_model(model_path: str, input_size: int, time_steps: int = 5):
    """Load a saved CNN model from disk"""
    model = StockCNN(input_size=input_size, time_steps=time_steps)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"CNN model loaded from {model_path}")
    return model


def evaluate_traditional_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate traditional model performance and return predictions.

    Returns:
        (results_dict, y_true, y_pred_proba)
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # positive class probability
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

    return results, y_test.values, y_pred_prob


def prepare_sequences(
    X: pd.DataFrame, y: pd.Series, time_steps: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare time series data for CNN"""
    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i : (i + time_steps)].values)
        y_seq.append(y.iloc[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


def evaluate_cnn_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, time_steps: int = 5
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate CNN model performance and return predictions.

    Returns:
        (results_dict, y_true, y_pred_proba)
    """
    X_seq, y_seq = prepare_sequences(X_test, y_test, time_steps)
    X_tensor = torch.FloatTensor(X_seq)

    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_tensor).numpy().flatten()

    y_pred = (y_pred_prob > 0.5).astype(int)

    results = {}
    for metric_name, metric_func in METRICS.items():
        try:
            if metric_name == "roc_auc":
                results[metric_name] = metric_func(y_seq, y_pred_prob)
            else:
                results[metric_name] = metric_func(y_seq, y_pred)
        except Exception as e:
            print(f"Error calculating {metric_name}: {str(e)}")
            results[metric_name] = 0.0

    return results, y_seq, y_pred_prob


def save_predictions(
    symbol: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: Path,
) -> str:
    """Save predictions to a numpy compressed file and return the file path."""
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    file_path = pred_dir / f"{symbol}_{model_name}_preds.npz"
    np.savez_compressed(file_path, y_true=y_true, y_pred_proba=y_pred_proba)
    return str(file_path)


def load_predictions(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from a numpy compressed file."""
    data = np.load(file_path)
    return data["y_true"], data["y_pred_proba"]


def get_traditional_feature_importance(
    model, feature_cols: List[str], model_type: str
) -> Dict[str, float]:
    """Get feature importance for traditional models"""
    if model_type == "logistic_regression":
        importance = model.coef_[0]
    elif model_type == "xgboost":
        importance = model.feature_importances_
    else:
        return {}

    # Normalize to 0-1 range
    importance = (importance - importance.min()) / (importance.max() - importance.min())
    return dict(zip(feature_cols, importance))


def get_cnn_feature_importance(
    model: nn.Module, feature_cols: List[str]
) -> Dict[str, float]:
    """Get CNN model feature importance"""
    conv1_weights = model.conv1.weight.data.numpy()
    feature_importance = np.mean(np.abs(conv1_weights), axis=(0, 2))
    feature_importance = (feature_importance - feature_importance.min()) / (
        feature_importance.max() - feature_importance.min()
    )
    return dict(zip(feature_cols, feature_importance))


def evaluate_single_stock_models(
    data_path: str,
    symbol: str,
    model_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Evaluate all models for a single stock and save predictions."""
    print(f"Evaluating all models for {symbol}")
    output_path = Path(output_dir)

    results = {
        "symbol": symbol,
        "logistic_regression": {},
        "xgboost": {},
        "cnn_specific": {},
        "cnn_universal": {},
        "feature_importance": {
            "logistic_regression": {},
            "xgboost": {},
            "cnn_specific": {},
            "cnn_universal": {},
        },
        "prediction_files": {},  # store paths to saved predictions
    }

    # Evaluate traditional models (universal features)
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data_path, symbol, "universal", test_size, random_state
        )

        results_path = Path(model_dir) / "ssm_uf_trad" / "training_results.json"
        with open(results_path, "r") as f:
            training_results = json.load(f)

        symbol_result = next(
            (r for r in training_results if r["symbol"] == symbol), None
        )
        if symbol_result is not None:
            feature_cols = symbol_result["feature_cols"]

            lr_model_path = (
                Path(model_dir)
                / "ssm_uf_trad"
                / "models"
                / f"{symbol}_logistic_regression.joblib"
            )
            if lr_model_path.exists():
                lr_model = load_traditional_model(str(lr_model_path))
                lr_results, y_true, y_pred_proba = evaluate_traditional_model(
                    lr_model, X_test, y_test
                )
                results["logistic_regression"] = lr_results
                results["feature_importance"]["logistic_regression"] = (
                    get_traditional_feature_importance(
                        lr_model, feature_cols, "logistic_regression"
                    )
                )
                results["prediction_files"]["logistic_regression"] = save_predictions(
                    symbol, "logistic_regression", y_true, y_pred_proba, output_path
                )

            xgb_model_path = (
                Path(model_dir) / "ssm_uf_trad" / "models" / f"{symbol}_xgboost.joblib"
            )
            if xgb_model_path.exists():
                xgb_model = load_traditional_model(str(xgb_model_path))
                xgb_results, y_true, y_pred_proba = evaluate_traditional_model(
                    xgb_model, X_test, y_test
                )
                results["xgboost"] = xgb_results
                results["feature_importance"]["xgboost"] = (
                    get_traditional_feature_importance(
                        xgb_model, feature_cols, "xgboost"
                    )
                )
                results["prediction_files"]["xgboost"] = save_predictions(
                    symbol, "xgboost", y_true, y_pred_proba, output_path
                )
    except Exception as e:
        print(f"Error evaluating traditional models for {symbol}: {str(e)}")

    # Evaluate CNN model with specific features
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data_path, symbol, "specific", test_size, random_state
        )

        results_path = Path(model_dir) / "ssm_sf_cnn" / "training_results.json"
        with open(results_path, "r") as f:
            training_results = json.load(f)

        symbol_result = next(
            (r for r in training_results if r["symbol"] == symbol), None
        )
        if symbol_result is not None:
            feature_cols = symbol_result["feature_cols"]
            time_steps = symbol_result["time_steps"]

            model_path = (
                Path(model_dir) / "ssm_sf_cnn" / "models" / f"{symbol}_ssm_sf_cnn.pth"
            )
            if model_path.exists():
                cnn_model = load_cnn_model(
                    str(model_path), len(feature_cols), time_steps
                )
                cnn_results, y_true, y_pred_proba = evaluate_cnn_model(
                    cnn_model, X_test, y_test, time_steps
                )
                results["cnn_specific"] = cnn_results
                results["feature_importance"]["cnn_specific"] = (
                    get_cnn_feature_importance(cnn_model, feature_cols)
                )
                results["prediction_files"]["cnn_specific"] = save_predictions(
                    symbol, "cnn_specific", y_true, y_pred_proba, output_path
                )
    except Exception as e:
        print(
            f"Error evaluating CNN model with specific features for {symbol}: {str(e)}"
        )

    # Evaluate CNN model with universal features
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data_path, symbol, "universal", test_size, random_state
        )

        results_path = Path(model_dir) / "ssm_uf_cnn" / "training_results.json"
        with open(results_path, "r") as f:
            training_results = json.load(f)

        symbol_result = next(
            (r for r in training_results if r["symbol"] == symbol), None
        )
        if symbol_result is not None:
            feature_cols = symbol_result["feature_cols"]
            time_steps = symbol_result["time_steps"]

            model_path = (
                Path(model_dir) / "ssm_uf_cnn" / "models" / f"{symbol}_ssm_uf_cnn.pth"
            )
            if model_path.exists():
                cnn_model = load_cnn_model(
                    str(model_path), len(feature_cols), time_steps
                )
                cnn_results, y_true, y_pred_proba = evaluate_cnn_model(
                    cnn_model, X_test, y_test, time_steps
                )
                results["cnn_universal"] = cnn_results
                results["feature_importance"]["cnn_universal"] = (
                    get_cnn_feature_importance(cnn_model, feature_cols)
                )
                results["prediction_files"]["cnn_universal"] = save_predictions(
                    symbol, "cnn_universal", y_true, y_pred_proba, output_path
                )
    except Exception as e:
        print(
            f"Error evaluating CNN model with universal features for {symbol}: {str(e)}"
        )

    return results


def plot_roc_curves_comparison(
    model_a: str,
    model_b: str,
    symbols: List[str],
    prediction_files: Dict[str, Dict[str, str]],
    output_path: Path,
) -> None:
    """
    Plot ROC curves for two models using aggregated predictions across all stocks.
    """
    plt.figure(figsize=(8, 6))

    # Aggregate predictions for each model
    all_true_a = []
    all_proba_a = []
    all_true_b = []
    all_proba_b = []

    for symbol in symbols:
        if model_a in prediction_files.get(symbol, {}):
            file_a = prediction_files[symbol][model_a]
            y_true_a, y_proba_a = load_predictions(file_a)
            all_true_a.extend(y_true_a)
            all_proba_a.extend(y_proba_a)

        if model_b in prediction_files.get(symbol, {}):
            file_b = prediction_files[symbol][model_b]
            y_true_b, y_proba_b = load_predictions(file_b)
            all_true_b.extend(y_true_b)
            all_proba_b.extend(y_proba_b)

    if all_true_a and all_proba_a:
        plot_single_roc_curve(all_true_a, all_proba_a, model_a)
    if all_true_b and all_proba_b:
        plot_single_roc_curve(all_true_b, all_proba_b, model_b)
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guessing"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {model_a} vs {model_b}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f"{model_a}_vs_{model_b}_roc_curves.png", dpi=300)
    plt.close()
    print(
        f"ROC curves saved to {output_path / f'{model_a}_vs_{model_b}_roc_curves.png'}"
    )


# TODO Rename this here and in `plot_roc_curves_comparison`
def plot_single_roc_curve(y_true, y_score, label_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{label_name} (AUC = {roc_auc:.3f})")


def create_comprehensive_visualizations(
    comparison_df: pd.DataFrame,
    prediction_files: Dict[str, Dict[str, str]],
    output_dir: str,
) -> None:
    """
    Create comprehensive visualizations including overall ROC curves.
    """
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (20, 18)

    fig, axes = plt.subplots(4, 3, figsize=(20, 18))
    fig.suptitle("Comprehensive Model Comparison", fontsize=16)

    # 1. Accuracy comparison by stock and model
    ax = axes[0, 0]
    pivot_acc = comparison_df.pivot(index="symbol", columns="model", values="accuracy")
    pivot_acc.plot(kind="bar", ax=ax)
    ax.set_title("Accuracy Comparison by Stock and Model")
    ax.set_xlabel("Stock Symbol")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)

    # 2. Average model performance
    ax = axes[0, 1]
    model_avg = comparison_df.groupby("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean()
    model_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Performance by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average Score")
    ax.legend(title="Metrics")
    ax.tick_params(axis="x", rotation=0)

    # 3. ROC-AUC heatmap
    ax = axes[0, 2]
    pivot_roc = comparison_df.pivot(index="symbol", columns="model", values="roc_auc")
    sns.heatmap(
        pivot_roc,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "ROC-AUC"},
        ax=ax,
    )
    ax.set_title("ROC-AUC Heatmap")

    # 4. Model stability (ROC-AUC distribution)
    ax = axes[1, 0]
    comparison_df.boxplot(column="roc_auc", by="model", ax=ax)
    ax.set_title("Stability of Models (ROC-AUC Distribution)")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    plt.suptitle("")

    # 5. Model type comparison
    ax = axes[1, 1]
    model_type_comparison = comparison_df.copy()
    model_type_comparison["model_type"] = model_type_comparison["model"].apply(
        lambda x: "CNN" if "cnn" in x else "Traditional"
    )
    model_type_avg = model_type_comparison.groupby("model_type")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean()
    model_type_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Performance by Model Type")
    ax.set_xlabel("Model Type")
    ax.set_ylabel("Average Score")
    ax.legend(title="Metrics")
    ax.tick_params(axis="x", rotation=0)

    # 6. Feature pipeline comparison
    ax = axes[1, 2]
    feature_pipeline_comparison = comparison_df.copy()
    feature_pipeline_comparison["feature_pipeline"] = feature_pipeline_comparison[
        "model"
    ].apply(lambda x: "Specific" if x == "cnn_specific" else "Universal")
    pipeline_avg = feature_pipeline_comparison.groupby("feature_pipeline")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean()
    pipeline_avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Performance by Feature Pipeline")
    ax.set_xlabel("Feature Pipeline")
    ax.set_ylabel("Average Score")
    ax.legend(title="Metrics")
    ax.tick_params(axis="x", rotation=0)

    # 7. Precision-Recall comparison
    ax = axes[2, 0]
    for model in comparison_df["model"].unique():
        model_data = comparison_df[comparison_df["model"] == model]
        ax.scatter(model_data["precision"], model_data["recall"], label=model, s=100)
    ax.set_title("Precision-Recall Comparison")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(True)

    # 8. F1 Score comparison
    ax = axes[2, 1]
    pivot_f1 = comparison_df.pivot(index="symbol", columns="model", values="f1")
    pivot_f1.plot(kind="bar", ax=ax)
    ax.set_title("F1 Score Comparison by Stock and Model")
    ax.set_xlabel("Stock Symbol")
    ax.set_ylabel("F1 Score")
    ax.legend(title="Model")
    ax.tick_params(axis="x", rotation=45)

    # 9. Overall model ranking
    ax = axes[2, 2]
    model_ranking = comparison_df.groupby("model").agg(
        {
            "accuracy": "mean",
            "precision": "mean",
            "recall": "mean",
            "f1": "mean",
            "roc_auc": "mean",
        }
    )
    model_ranking["overall_score"] = (
        model_ranking["accuracy"] * 0.2
        + model_ranking["precision"] * 0.2
        + model_ranking["recall"] * 0.2
        + model_ranking["f1"] * 0.2
        + model_ranking["roc_auc"] * 0.2
    )
    model_ranking = model_ranking.sort_values("overall_score", ascending=False)
    model_ranking["overall_score"].plot(kind="barh", ax=ax)
    ax.set_title("Overall Model Ranking")
    ax.set_xlabel("Overall Score")
    ax.set_ylabel("Model")

    # 10. ROC Curves for all models (aggregated across stocks)
    ax = axes[3, 0]
    models_to_plot = comparison_df["model"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
    for model, color in zip(models_to_plot, colors):
        all_true = []
        all_proba = []
        for symbol in comparison_df["symbol"].unique():
            if symbol in prediction_files and model in prediction_files[symbol]:
                y_true, y_proba = load_predictions(prediction_files[symbol][model])
                all_true.extend(y_true)
                all_proba.extend(y_proba)
        if all_true and all_proba:
            fpr, tpr, _ = roc_curve(all_true, all_proba)
            roc_auc_val = auc(fpr, tpr)
            ax.plot(
                fpr, tpr, lw=2, color=color, label=f"{model} (AUC={roc_auc_val:.3f})"
            )
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Guessing")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for All Models (Aggregated)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # 11. Empty subplot to fill layout
    axes[3, 1].axis("off")
    axes[3, 2].axis("off")

    plt.tight_layout()
    # Save to output_dir (the main output directory)
    output_path = Path(output_dir)
    plt.savefig(output_path / "comprehensive_visualization.png", dpi=300)
    plt.show()


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """
    Compare results across different stocks and models with enhanced visualizations.
    """
    comparison_data = []

    for result in results:
        symbol = result["symbol"]
        for model_name, metrics in result.items():
            if (
                model_name
                in [
                    "logistic_regression",
                    "xgboost",
                    "cnn_specific",
                    "cnn_universal",
                ]
                and metrics
            ):
                row = {"symbol": symbol, "model": model_name}
                row |= metrics
                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

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

    print("\nAverage ROC-AUC by Model Type:")
    model_type_comparison = comparison_df.copy()
    model_type_comparison["model_type"] = model_type_comparison["model"].apply(
        lambda x: "CNN" if "cnn" in x else "Traditional"
    )
    for model_type in model_type_comparison["model_type"].unique():
        avg_roc = model_type_comparison[
            model_type_comparison["model_type"] == model_type
        ]["roc_auc"].mean()
        print(f"  {model_type}: {avg_roc:.4f}")

    print("\nAverage ROC-AUC by Feature Pipeline:")
    feature_pipeline_comparison = comparison_df.copy()
    feature_pipeline_comparison["feature_pipeline"] = feature_pipeline_comparison[
        "model"
    ].apply(lambda x: "Specific" if x == "cnn_specific" else "Universal")
    for pipeline in feature_pipeline_comparison["feature_pipeline"].unique():
        avg_roc = feature_pipeline_comparison[
            feature_pipeline_comparison["feature_pipeline"] == pipeline
        ]["roc_auc"].mean()
        print(f"  {pipeline}: {avg_roc:.4f}")

    return comparison_df


def save_results(
    results: List[Dict], output_dir: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Save experiment results and return comparison DataFrame and prediction file mapping."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save comparison table
    comparison_df = compare_models(results)
    comparison_df.to_csv(output_path / "model_comparison.csv", index=False)

    # Save detailed results for each stock
    prediction_files = {}
    for result in results:
        symbol = result["symbol"]
        symbol_path = output_path / f"{symbol}_evaluation_results.json"

        # Save only non-prediction data in JSON
        result_copy = result.copy()
        if "prediction_files" in result_copy:
            prediction_files[symbol] = result_copy["prediction_files"]
            del result_copy["prediction_files"]  # remove from JSON

        with open(symbol_path, "w") as f:
            json.dump(result_copy, f, indent=2, default=str)

    print(f"Results saved to {output_path}")
    return comparison_df, prediction_files


def generate_conclusions(comparison_df: pd.DataFrame) -> Dict:
    """Generate experiment conclusions"""
    best_roc = comparison_df.loc[comparison_df["roc_auc"].idxmax()]
    model_stability = comparison_df.groupby("model")["roc_auc"].std().to_dict()
    stock_modelability = comparison_df.groupby("symbol")["roc_auc"].mean().to_dict()

    model_type_comparison = comparison_df.copy()
    model_type_comparison["model_type"] = model_type_comparison["model"].apply(
        lambda x: "CNN" if "cnn" in x else "Traditional"
    )
    model_type_performance = (
        model_type_comparison.groupby("model_type")["roc_auc"].mean().to_dict()
    )

    feature_pipeline_comparison = comparison_df.copy()
    feature_pipeline_comparison["feature_pipeline"] = feature_pipeline_comparison[
        "model"
    ].apply(lambda x: "Specific" if x == "cnn_specific" else "Universal")
    pipeline_performance = (
        feature_pipeline_comparison.groupby("feature_pipeline")["roc_auc"]
        .mean()
        .to_dict()
    )

    models = list(model_stability.keys())
    recommended_model = min(
        models,
        key=lambda m: (
            model_stability[m],
            -comparison_df[comparison_df["model"] == m]["roc_auc"].mean(),
        ),
    )

    recommended_model_type = max(model_type_performance, key=model_type_performance.get)
    recommended_pipeline = max(pipeline_performance, key=pipeline_performance.get)

    return {
        "best_performing": {
            "symbol": best_roc["symbol"],
            "model": best_roc["model"],
            "roc_auc": best_roc["roc_auc"],
        },
        "model_stability": model_stability,
        "stock_modelability": stock_modelability,
        "model_type_performance": model_type_performance,
        "pipeline_performance": pipeline_performance,
        "recommendations": {
            "recommended_symbol": max(stock_modelability, key=stock_modelability.get),
            "recommended_model": recommended_model,
            "recommended_model_type": recommended_model_type,
            "recommended_pipeline": recommended_pipeline,
        },
    }


def preprocess_model_data(results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Preprocess data for each model to facilitate comparison."""
    preprocessed_data = {
        "logistic_regression": pd.DataFrame(),
        "xgboost": pd.DataFrame(),
        "cnn_specific": pd.DataFrame(),
        "cnn_universal": pd.DataFrame(),
    }

    for result in results:
        symbol = result["symbol"]
        for model_name in preprocessed_data:
            if model_name in result and result[model_name]:
                row = {"symbol": symbol}
                row |= result[model_name]
                preprocessed_data[model_name] = pd.concat(
                    [preprocessed_data[model_name], pd.DataFrame([row])],
                    ignore_index=True,
                )

    return preprocessed_data


def compare_two_models(
    model_a: str,
    model_b: str,
    preprocessed_data: Dict[str, pd.DataFrame],
    prediction_files: Dict[str, Dict[str, str]],
    symbols: List[str],
    output_dir: str = "outputs/all_models/comparisons",
) -> None:
    """Compare two models using ROC curves instead of bar charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot ROC curves (replaces the previous bar chart)
    plot_roc_curves_comparison(model_a, model_b, symbols, prediction_files, output_path)


def main() -> None:
    args = parse_args()

    # Evaluate models for all stocks
    all_results = []
    for symbol in args.symbols:
        result = evaluate_single_stock_models(
            args.data_path,
            symbol,
            args.model_dir,
            args.output_dir,
            args.test_size,
            args.random_state,
        )
        if result is not None:
            all_results.append(result)

    # Save results and get prediction file mapping
    comparison_df, prediction_files = save_results(all_results, args.output_dir)

    # Generate conclusions
    conclusions = generate_conclusions(comparison_df)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "conclusions.json", "w") as f:
        json.dump(conclusions, f, indent=2, default=str)

    print("\nExperiment Conclusions:")
    print(f"Best Performance: {conclusions['best_performing']}")
    print(f"Model Stability: {conclusions['model_stability']}")
    print(f"Stock Modelability: {conclusions['stock_modelability']}")
    print(f"Model Type Performance: {conclusions['model_type_performance']}")
    print(f"Pipeline Performance: {conclusions['pipeline_performance']}")
    print(f"Recommendations: {conclusions['recommendations']}")

    # Create comprehensive visualizations including ROC curves
    create_comprehensive_visualizations(
        comparison_df, prediction_files, args.output_dir
    )

    # Preprocess data for model comparison
    preprocessed_data = preprocess_model_data(all_results)

    # Example: Compare two models using ROC curves
    model_a = "logistic_regression"
    model_b = "xgboost"
    compare_two_models(
        model_a,
        model_b,
        preprocessed_data,
        prediction_files,
        args.symbols,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
