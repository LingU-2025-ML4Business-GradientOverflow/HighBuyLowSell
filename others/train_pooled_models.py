import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


from data import load_stock_data
from feature_pipeline_universal import feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pooled issue3B baseline models."
    )
    parser.add_argument(
        "--input",
        default="data/raw/yahoo_daily_prices.csv",
        help="Path to raw stock CSV data.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Chronological train split ratio.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/issue3b",
        help="Directory for scenario metrics and predictions.",
    )
    parser.add_argument(
        "--feature-output",
        default="data/processed/issue3b_feature_table.csv",
        help="Path to save the pooled feature table.",
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
            "scenario_name": "pooled_numeric_only_logistic_regression",
            "display_name": "numeric / logistic",
            "feature_set": "numeric_only",
            "model_name": "logistic_regression",
        },
        {
            "scenario_name": "pooled_numeric_only_xgboost",
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


def split_feature_table(
    feature_table: pd.DataFrame, split_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = feature_table.sort_values(["date", "symbol"]).reset_index(drop=True)
    split_index = int(len(ordered) * split_ratio)
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


def run_scenarios(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    scenarios = build_scenarios()
    models, notes = build_model_factories()

    scenario_rows: list[dict[str, object]] = []
    per_symbol_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    train_inputs = train_frame[["symbol"] + features_list]
    test_inputs = test_frame[["symbol"] + features_list]
    y_train = train_frame["target"]

    total_scenarios = len(scenarios)

    for index, scenario in enumerate(scenarios, start=1):
        scenario_start = time.perf_counter()
        print("")
        print(f"[S{index}/{total_scenarios}] {scenario['display_name']}")
        print(f"  key:   {scenario['scenario_name']}")
        print(f"  input: {scenario['feature_set']} + {scenario['model_name']}")
        estimator = build_estimator(
            scenario["model_name"], scenario["feature_set"], models
        )
        estimator.fit(train_inputs, y_train)

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
        elapsed = round(time.perf_counter() - scenario_start, 2)
        print("  metrics:")
        print(f"    accuracy:               {summary_metrics['accuracy']}")
        print(f"    f1:                     {summary_metrics['f1']}")
        print(f"    roc_auc:                {summary_metrics['roc_auc']}")
        print(
            f"    prediction_pos_ratio:   {summary_metrics['prediction_positive_ratio']}"
        )
        print(f"  time_s: {elapsed}")

        for symbol, symbol_frame in scenario_predictions.groupby("symbol", sort=True):
            symbol_metrics = score_partition(symbol_frame)
            per_symbol_rows.append(
                {
                    "scenario_name": scenario["scenario_name"],
                    "feature_set": scenario["feature_set"],
                    "model_name": scenario["model_name"],
                    "symbol": symbol,
                    **symbol_metrics,
                }
            )

    scenario_metrics = pd.DataFrame(scenario_rows)
    per_symbol_metrics = pd.DataFrame(per_symbol_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return scenario_metrics, per_symbol_metrics, predictions, notes


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()

    print("[Issue3B] Pooled training")
    print("")
    print("[1/4] Loading raw data")
    print(f"  input: {args.input}")
    raw_data = load_stock_data(args.input)
    print(f"  rows:   {len(raw_data)}")
    print(f"  symbols:{raw_data['symbol'].nunique()}")

    print("")
    print("[2/4] Building feature table")
    feature_table = feature_pipeline.fit_transform(raw_data)
    feature_table = feature_table.replace([np.inf, -np.inf], pd.NA)
    feature_table = feature_table.dropna(
        subset=features_list + ["return_1d", "target"]
    ).reset_index(drop=True)
    print(f"  rows:             {len(feature_table)}")
    print(f"  numeric_features: {len(features_list)}")

    feature_output = Path(args.feature_output)
    feature_output.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(feature_output, index=False)
    print(f"  saved:            {feature_output}")

    train_frame, test_frame = split_feature_table(feature_table, args.split_ratio)
    print("")
    print("[3/4] Chronological split")
    print(f"  split_ratio: {args.split_ratio}")
    print(f"  train_rows:  {len(train_frame)}")
    print(f"  test_rows:   {len(test_frame)}")

    print("")
    print("[4/4] Running scenarios")
    scenario_metrics, per_symbol_metrics, predictions, notes = run_scenarios(
        train_frame, test_frame
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_metrics_path = output_dir / "scenario_metrics.csv"
    per_symbol_metrics_path = output_dir / "per_symbol_metrics.csv"
    predictions_path = output_dir / "predictions.csv"
    notes_path = output_dir / "notes.txt"

    scenario_metrics.to_csv(scenario_metrics_path, index=False)
    per_symbol_metrics.to_csv(per_symbol_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    if notes:
        notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
    elif notes_path.exists():
        notes_path.unlink()

    print("")
    print("[Done] Saved outputs")
    print(f"  features:           {feature_output}")
    print(f"  scenario_metrics:   {scenario_metrics_path}")
    print(f"  per_symbol_metrics: {per_symbol_metrics_path}")
    print(f"  predictions:        {predictions_path}")
    if notes:
        print(f"  notes:              {notes_path}")
    print(f"  total_time_s:       {round(time.perf_counter() - total_start, 2)}")


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
