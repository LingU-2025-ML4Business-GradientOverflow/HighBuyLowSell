import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

from data import load_stock_data
from feature_pipeline_universal import feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train single stock models using traditional ML methods."
    )
    parser.add_argument(
        "--input",
        default="data/raw/yahoo_daily_prices.csv",
        help="Path to raw stock CSV data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set ratio (chronological split).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/psm_uf_trad",
        help="Directory for scenario metrics and predictions.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="List of stock symbols to model. If None, models all symbols found.",
    )
    return parser.parse_args()


def safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def rounded_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def build_scenarios() -> list[dict[str, str]]:
    return [
        {
            "scenario_name": "psm_logistic_regression",
            "display_name": "numeric / logistic",
            "feature_set": "numeric_only",
            "model_name": "logistic_regression",
        },
        {
            "scenario_name": "psm_xgboost",
            "display_name": "numeric / xgboost",
            "feature_set": "numeric_only",
            "model_name": "xgboost",
        },
    ]


def build_model_factories() -> tuple[dict[str, object], list[str]]:
    notes: list[str] = []
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
    }

    models["xgboost"] = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )

    return models, notes


def build_preprocessor(
    feature_set: str, numeric_columns: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", StandardScaler(), numeric_columns),
    ]

    return ColumnTransformer(transformers=transformers)


def build_estimator(
    model_name: str, feature_set: str, models: dict[str, object]
) -> Pipeline:
    estimator = clone(models[model_name])
    preprocessor = build_preprocessor(feature_set, features_list)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def split_data(
    data: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = data.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Split ratio produced an empty train or test partition.")
    return ordered.iloc[:split_index].copy(), ordered.iloc[split_index:].copy()


def score_partition(partition: pd.DataFrame) -> dict[str, float | None]:
    roc_auc = safe_roc_auc(partition["target"], partition["prediction_probability"])
    return {
        "accuracy": rounded_metric(
            accuracy_score(partition["target"], partition["predicted_label"])
        ),
        "f1": rounded_metric(
            f1_score(partition["target"], partition["predicted_label"])
        ),
        "roc_auc": rounded_metric(roc_auc),
        "prediction_positive_ratio": rounded_metric(
            partition["predicted_label"].mean()
        ),
        "test_rows": int(len(partition)),
    }


def run_single_stock_scenarios(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    symbol: str,
    output_dir: Path,  # 添加output_dir参数
) -> tuple[pd.DataFrame, list[str]]:
    scenarios = build_scenarios()
    models, notes = build_model_factories()

    scenario_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    # 创建模型保存目录
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = train_frame[features_list]
    test_inputs = test_frame[features_list]
    y_train = train_frame["target"]

    total_scenarios = len(scenarios)

    for index, scenario in enumerate(scenarios, start=1):
        print(f"  [{symbol}] [S{index}/{total_scenarios}] {scenario['display_name']}")

        estimator = build_estimator(
            scenario["model_name"], scenario["feature_set"], models
        )
        estimator.fit(train_inputs, y_train)

        # 保存训练好的模型
        model_path = models_dir / f"{symbol}_{scenario['model_name']}.joblib"
        joblib.dump(estimator, model_path)
        print(f"    Model saved to {model_path}")

        probabilities = estimator.predict_proba(test_inputs)[:, 1]
        predicted_labels = (probabilities >= 0.5).astype(int)

        scenario_predictions = test_frame[
            ["date", "symbol", "close", "return_1d", "target"]
        ].copy()
        scenario_predictions["scenario_name"] = scenario["scenario_name"]
        scenario_predictions["feature_set"] = scenario["feature_set"]
        scenario_predictions["model_name"] = scenario["model_name"]
        scenario_predictions["prediction_probability"] = probabilities
        scenario_predictions["predicted_label"] = predicted_labels
        prediction_frames.append(scenario_predictions)

        summary_metrics = score_partition(scenario_predictions)
        scenario_rows.append(
            {
                "symbol": symbol,
                "scenario_name": scenario["scenario_name"],
                "feature_set": scenario["feature_set"],
                "model_name": scenario["model_name"],
                "feature_count": int(
                    estimator.named_steps["preprocessor"]
                    .get_feature_names_out()
                    .shape[0]
                ),
                "train_rows": int(len(train_frame)),
                **summary_metrics,
            }
        )

    scenario_metrics = pd.DataFrame(scenario_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return scenario_metrics, predictions, notes


def main() -> None:
    args = parse_args()

    print("[SSM-UF-Trad] Single Stock Modeling")
    print("")
    print("[1/3] Loading raw data")
    print(f"  input: {args.input}")
    raw_data = load_stock_data(args.input)

    if args.symbols:
        symbols = args.symbols
    else:
        symbols = sorted(raw_data["symbol"].unique().tolist())

    print(f"  total symbols: {raw_data['symbol'].nunique()}")
    print(f"  modeling symbols: {len(symbols)}")

    print("")
    print("[2/3] Building feature table")
    feature_table = feature_pipeline.fit_transform(raw_data)
    feature_table = feature_table.replace([np.inf, -np.inf], pd.NA)
    feature_table = feature_table.dropna(
        subset=features_list + ["return_1d", "target"]
    ).reset_index(drop=True)
    print(f"  rows: {len(feature_table)}")
    print(f"  numeric_features: {len(features_list)}")

    all_scenario_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_notes: list[str] = []

    print("")
    print("[3/3] Running scenarios per symbol")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        symbol_data = feature_table[feature_table["symbol"] == symbol].copy()
        if len(symbol_data) < 50:
            print(f"  [{symbol}] Skipped (insufficient data: {len(symbol_data)} rows)")
            continue

        train_frame, test_frame = split_data(symbol_data, args.test_size)

        scenario_metrics, predictions, notes = run_single_stock_scenarios(
            train_frame, test_frame, symbol, output_dir
        )

        all_scenario_metrics.append(scenario_metrics)
        all_predictions.append(predictions)
        all_notes.extend(notes)

    if all_scenario_metrics:
        combined_metrics = pd.concat(all_scenario_metrics, ignore_index=True)
        metrics_path = output_dir / "scenario_metrics.csv"
        combined_metrics.to_csv(metrics_path, index=False)
        print(f"  scenario_metrics: {metrics_path}")

    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        predictions_path = output_dir / "predictions.csv"
        combined_predictions.to_csv(predictions_path, index=False)
        print(f"  predictions: {predictions_path}")

    if all_notes:
        notes_path = output_dir / "notes.txt"
        notes_path.write_text("\n".join(all_notes) + "\n", encoding="utf-8")
        print(f"  notes: {notes_path}")


features_list = [
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_mid",
    "bb_std",
    "bb_upper",
    "bb_lower",
    "vol_price_divergence",
    "return_1d",
    "ma_gap_10",
    "volatility_5d",
]


if __name__ == "__main__":
    main()
