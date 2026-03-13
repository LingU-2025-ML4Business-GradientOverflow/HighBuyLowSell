from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = str(exc)

from data import load_stock_data, select_symbol
from features import build_feature_table, feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline stock direction models.")
    parser.add_argument("--input", required=True, help="Path to raw stock CSV data.")
    parser.add_argument("--symbol", default=None, help="Single symbol to model.")
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for model outputs and metrics.",
    )
    parser.add_argument(
        "--feature-output",
        default="data/processed/feature_table.csv",
        help="Path to save the feature table.",
    )
    return parser.parse_args()


def build_models() -> tuple[dict[str, object], list[str]]:
    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
    }
    notes: list[str] = []

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        models["hist_gradient_boosting_fallback"] = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            random_state=42,
        )
        notes.append(
            "XGBoost unavailable in this environment; used HistGradientBoostingClassifier as fallback."
        )
        notes.append(f"XGBoost import error: {XGBOOST_IMPORT_ERROR}")

    return models, notes


def main() -> None:
    args = parse_args()

    raw_data = load_stock_data(args.input)
    scoped_data = select_symbol(raw_data, args.symbol)
    feature_table = build_feature_table(scoped_data)

    feature_output = Path(args.feature_output)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(feature_output, index=False)

    X = feature_table[feature_columns()]
    y = feature_table["target_direction"]

    split_index = int(len(feature_table) * args.split_ratio)
    if split_index <= 0 or split_index >= len(feature_table):
        raise ValueError("Split ratio produced an empty train or test partition.")

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    test_metadata = feature_table.iloc[split_index:][["date", "symbol", "close", "future_return_1d"]].copy()

    models, notes = build_models()
    metrics: dict[str, dict[str, float] | list[str]] = {"notes": notes}
    predictions = test_metadata.reset_index(drop=True)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(int)

        metrics[model_name] = {
            "accuracy": round(accuracy_score(y_test, predicted_labels), 4),
            "f1": round(f1_score(y_test, predicted_labels), 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        }

        predictions[f"{model_name}_probability"] = probabilities
        predictions[f"{model_name}_prediction"] = predicted_labels

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    predictions.to_csv(predictions_path, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved features to {feature_output}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
