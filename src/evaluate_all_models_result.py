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

    threshold = 0.3
    y_pred = (y_pred_prob >= threshold).astype(int)
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
        "psm_logistic_regression": {},
        "psm_xgboost": {},
        "feature_importance": {
            "logistic_regression": {},
            "xgboost": {},
            "cnn_specific": {},
            "cnn_universal": {},
            "psm_logistic_regression": {},
            "psm_xgboost": {},
        },
        "prediction_files": {},
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

    # Evaluate PSM models universal features
    try:
        X_train, X_test, y_train, y_test = prepare_data(
            data_path, symbol, "universal", test_size, random_state
        )

        results_path = Path(model_dir) / "psm_uf_trad" / "scenario_metrics.csv"
        if results_path.exists():
            psm_results = pd.read_csv(results_path)

            symbol_psm_results = psm_results[psm_results["symbol"] == symbol]

            for _, row in symbol_psm_results.iterrows():
                model_name = row["model_name"]

                model_path = (
                    Path(model_dir)
                    / "psm_uf_trad"
                    / "models"
                    / f"{symbol}_{model_name}.joblib"
                )

                if model_path.exists():
                    model = load_traditional_model(str(model_path))
                    model_results, y_true, y_pred_proba = evaluate_traditional_model(
                        model, X_test, y_test
                    )

                    psm_key = f"psm_{model_name}"
                    results[psm_key] = model_results

                    feature_cols = X_test.columns.tolist()
                    actual_model = (
                        model.named_steps["model"]
                        if hasattr(model, "named_steps")
                        else model
                    )
                    results["feature_importance"][psm_key] = (
                        get_traditional_feature_importance(
                            actual_model, feature_cols, model_name
                        )
                    )

                    results["prediction_files"][psm_key] = save_predictions(
                        symbol, psm_key, y_true, y_pred_proba, output_path
                    )
    except Exception as e:
        print(f"Error evaluating PSM models for {symbol}: {str(e)}")

    return results


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
                    "psm_logistic_regression",
                    "psm_xgboost",
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
    ].apply(lambda x: "Universal" if "universal" in x else "Specific")
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
    (pd.DataFrame(results)).to_csv(output_path / "all_results.csv", index=False)
    results_copy = results.copy()
    for result in results_copy:
        for k, v in result["feature_importance"].items():
            if isinstance(v, np.ndarray):
                result["feature_importance"][k] = v.tolist()
    with open(output_path / "all_results.json", "w") as f:
        json.dump(results_copy, f, indent=2, default=str)

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


def preprocess_model_data(results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Preprocess data for each model to facilitate comparison."""
    preprocessed_data = {
        "logistic_regression": pd.DataFrame(),
        "xgboost": pd.DataFrame(),
        "cnn_specific": pd.DataFrame(),
        "cnn_universal": pd.DataFrame(),
        "psm_logistic_regression": pd.DataFrame(),
        "psm_xgboost": pd.DataFrame(),
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


def predict_with_trained_models(
    data_path: str,
    symbols: List[str],
    model_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Use trained models to predict on the last 20% of data and save results.

    Args:
        data_path: Path to the data file
        symbols: List of stock symbols to predict
        model_dir: Directory containing trained models
        output_dir: Directory to save prediction results
        test_size: Test set ratio (default: 0.2)
        random_state: Random seed
    """
    output_path = Path(output_dir)
    prediction_output_dir = output_path / "future_predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    # Load full data
    full_data = pd.read_csv(data_path)

    for symbol in symbols:
        print(f"Generating predictions for {symbol}")

        # Filter specific stock data
        stock_data = full_data[full_data["symbol"] == symbol].copy()

        # Process data using universal feature pipeline
        processed_data = feature_pipeline_universal.fit_transform(stock_data)

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

        # Split data: first 80% for training, last 20% for prediction
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_future = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_future = y.iloc[:split_idx], y.iloc[split_idx:]

        # Load and use traditional models
        for model_name in ["logistic_regression", "xgboost"]:
            model_path = (
                Path(model_dir)
                / "ssm_uf_trad"
                / "models"
                / f"{symbol}_{model_name}.joblib"
            )

            if model_path.exists():
                try:
                    # Load model
                    model = load_traditional_model(str(model_path))

                    # Make predictions
                    y_pred_proba = model.predict_proba(X_future)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    # Save predictions
                    output_file = (
                        prediction_output_dir
                        / f"{symbol}_{model_name}_future_predictions.npz"
                    )
                    np.savez_compressed(
                        output_file,
                        dates=processed_data.iloc[split_idx:]["date"].values,
                        y_true=y_future.values,
                        y_pred=y_pred,
                        y_pred_proba=y_pred_proba,
                    )
                    print(f"Saved {model_name} predictions to {output_file}")
                except Exception as e:
                    print(f"Error predicting with {model_name} for {symbol}: {str(e)}")

        # Load and use pooled traditional models
        for model_name in ["psm_logistic_regression", "psm_xgboost"]:
            model_path = (
                Path(model_dir)
                / "psm_uf_trad"
                / "models"
                / f"{symbol}_{model_name}.joblib"
            )

            if model_path.exists():
                try:
                    # Load model
                    model = load_traditional_model(str(model_path))

                    # Make predictions
                    y_pred_proba = model.predict_proba(X_future)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    # Save predictions
                    output_file = (
                        prediction_output_dir
                        / f"{symbol}_{model_name}_future_predictions.npz"
                    )
                    np.savez_compressed(
                        output_file,
                        dates=processed_data.iloc[split_idx:]["date"].values,
                        y_true=y_future.values,
                        y_pred=y_pred,
                        y_pred_proba=y_pred_proba,
                    )
                    print(f"Saved {model_name} predictions to {output_file}")
                except Exception as e:
                    print(f"Error predicting with {model_name} for {symbol}: {str(e)}")

        # Load and use CNN model with universal features
        cnn_uf_model_path = (
            Path(model_dir) / "ssm_uf_cnn" / "models" / f"{symbol}_ssm_uf_cnn.pth"
        )

        if cnn_uf_model_path.exists():
            try:
                # Load training results to get time_steps
                results_path = Path(model_dir) / "ssm_uf_cnn" / "training_results.json"
                with open(results_path, "r") as f:
                    training_results = json.load(f)

                symbol_result = next(
                    (r for r in training_results if r["symbol"] == symbol), None
                )

                if symbol_result is not None:
                    time_steps = symbol_result["time_steps"]

                    # Load model
                    cnn_model = load_cnn_model(
                        str(cnn_uf_model_path), len(feature_cols), time_steps
                    )

                    # Prepare sequences for prediction
                    # We need to include some data from the training set to create sequences
                    X_for_seq = pd.concat([X_train.iloc[-time_steps:], X_future])
                    X_seq, _ = prepare_sequences(X_for_seq, y, time_steps)

                    # Make predictions
                    X_tensor = torch.FloatTensor(X_seq)
                    cnn_model.eval()
                    with torch.no_grad():
                        y_pred_proba = cnn_model(X_tensor).numpy().flatten()

                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    # Save predictions
                    output_file = (
                        prediction_output_dir
                        / f"{symbol}_cnn_universal_future_predictions.npz"
                    )
                    np.savez_compressed(
                        output_file,
                        dates=processed_data.iloc[split_idx:]["date"].values,
                        y_true=y_future.values,
                        y_pred=y_pred,
                        y_pred_proba=y_pred_proba,
                    )
                    print(f"Saved CNN universal predictions to {output_file}")
            except Exception as e:
                print(f"Error predicting with CNN universal for {symbol}: {str(e)}")

        cnn_sf_model_path = (
            Path(model_dir) / "ssm_uf_cnn" / "models" / f"{symbol}_ssm_uf_cnn.pth"
        )

        if cnn_sf_model_path.exists():
            try:
                # Load training results to get time_steps
                results_path = Path(model_dir) / "ssm_uf_cnn" / "training_results.json"
                with open(results_path, "r") as f:
                    training_results = json.load(f)

                symbol_result = next(
                    (r for r in training_results if r["symbol"] == symbol), None
                )

                if symbol_result is not None:
                    time_steps = symbol_result["time_steps"]

                    # Load model
                    cnn_model = load_cnn_model(
                        str(cnn_sf_model_path), len(feature_cols), time_steps
                    )

                    # Prepare sequences for prediction
                    # We need to include some data from the training set to create sequences
                    X_for_seq = pd.concat([X_train.iloc[-time_steps:], X_future])
                    X_seq, _ = prepare_sequences(X_for_seq, y, time_steps)

                    # Make predictions
                    X_tensor = torch.FloatTensor(X_seq)
                    cnn_model.eval()
                    with torch.no_grad():
                        y_pred_proba = cnn_model(X_tensor).numpy().flatten()

                    y_pred = (y_pred_proba >= 0.5).astype(int)

                    # Save predictions
                    output_file = (
                        prediction_output_dir
                        / f"{symbol}_cnn_specific_future_predictions.npz"
                    )
                    np.savez_compressed(
                        output_file,
                        dates=processed_data.iloc[split_idx:]["date"].values,
                        y_true=y_future.values,
                        y_pred=y_pred,
                        y_pred_proba=y_pred_proba,
                    )
                    print(f"Saved CNN stock future predictions to {output_file}")
            except Exception as e:
                print(f"Error predicting with CNN stock future for {symbol}: {str(e)}")


def main() -> None:
    args = {
        "data_path": "data/raw/yahoo_daily_prices.csv",
        "symbols": [
            "0700.HK",
            "BABA",
            "BIDU",
            "GOOGL",
            "MSFT",
            "NVDA",
        ],
        "model_dir": "outputs",
        "output_dir": "outputs/all_models",
        "test_size": 0.2,
        "random_state": 42,
    }
    # Evaluate models for all stocks
    all_results = []
    for symbol in args["symbols"]:
        result = evaluate_single_stock_models(
            args["data_path"],
            symbol,
            args["model_dir"],
            args["output_dir"],
            args["test_size"],
            args["random_state"],
        )
        if result is not None:
            all_results.append(result)

    # Save results and get prediction file mapping
    save_results(all_results, args["output_dir"])

    # Add predictions for the last 20% of data
    predict_with_trained_models(
        args["data_path"],
        args["symbols"],
        args["model_dir"],
        args["output_dir"],
        args["test_size"],
        args["random_state"],
    )


if __name__ == "__main__":
    main()
