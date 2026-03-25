import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import seaborn as sns
import csv

RESULT_FILE = "./outputs/all_models/model_comparison.csv"
ALL_RESULT = "./outputs/all_models/all_results.json"

p = []
with open(RESULT_FILE, encoding="utf-8") as f:
    reader_obj = csv.reader(f)
    p.extend(iter(reader_obj))

with open(ALL_RESULT, "r", encoding="utf-8") as f:
    results = json.load(f)

comparison_df = pd.read_csv(RESULT_FILE)


def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


prediction_files = {}
for result in results:
    symbol = result["symbol"]
    symbol_path = f"./outputs/all_models/{symbol}_evaluation_results.json"

    # Save only non-prediction data in JSON
    result_copy = result.copy()
    if "prediction_files" in result_copy:
        prediction_files[symbol] = result_copy["prediction_files"]
        del result_copy["prediction_files"]  # remove from JSON

    with open(symbol_path, "w") as f:
        json.dump(result_copy, f, indent=2, default=str)


def load_predictions(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions from a numpy compressed file."""
    data = np.load(file_path)
    return data["y_true"], data["y_pred_proba"]


width = 0.1

# Single stock LG vs XG
for _ in [0]:
    ss_lg_arr = [i for i in p if i[1] == "logistic_regression"]
    ss_lg_acc_avg = np.mean([float(i[2]) for i in ss_lg_arr])
    ss_lg_pre_avg = np.mean([float(i[3]) for i in ss_lg_arr])
    ss_lg_rec_avg = np.mean([float(i[4]) for i in ss_lg_arr])
    ss_lg_f1__avg = np.mean([float(i[5]) for i in ss_lg_arr])

    ss_xg_arr = [i for i in p if i[1] == "xgboost"]
    ss_xg_acc_avg = np.mean([float(i[2]) for i in ss_xg_arr])
    ss_xg_pre_avg = np.mean([float(i[3]) for i in ss_xg_arr])
    ss_xg_rec_avg = np.mean([float(i[4]) for i in ss_xg_arr])
    ss_xg_f1__avg = np.mean([float(i[5]) for i in ss_xg_arr])

    x = np.arange(2)
    x_acc = x - width * 1.5
    x_pre = x - width / 2
    x_rec = x + width / 2
    x_f1_ = x + width * 1.5

    fig, ax = plt.subplots(figsize=(10, 6))
    b_acc = ax.bar(
        x_acc,
        [ss_lg_acc_avg, ss_xg_acc_avg],
        width=width,
        label="accuracy",
        color="#f9766e",
        edgecolor="k",
        zorder=2,
    )
    b_pre = ax.bar(
        x_pre,
        [ss_lg_pre_avg, ss_xg_pre_avg],
        width=width,
        label="precision",
        color="#cccc11",
        edgecolor="k",
        zorder=2,
    )
    b_rec = ax.bar(
        x_rec,
        [ss_lg_rec_avg, ss_xg_rec_avg],
        width=width,
        label="recall",
        color="#66ccff",
        edgecolor="k",
        zorder=2,
    )
    b_f1_ = ax.bar(
        x_f1_,
        [ss_lg_f1__avg, ss_xg_f1__avg],
        width=width,
        label="f1",
        color="#39c5bb",
        edgecolor="k",
        zorder=2,
    )

    add_value_labels(ax, b_acc)
    add_value_labels(ax, b_pre)
    add_value_labels(ax, b_rec)
    add_value_labels(ax, b_f1_)

    ax.set_xticks(x)
    ax.set_xticklabels(["Logistic Regression", "XGBoost"])
    ax.legend()
    ax.set_title("Single Stock Logistic Regression vs XGBoost")
    fig.savefig("./outputs/all_models/ss_lg_xg.png")
    plt.close()

# Pooled stock LG vs XG
for _ in [0]:
    ps_lg_arr = [i for i in p if i[1] == "psm_logistic_regression"]
    ps_lg_acc_avg = np.mean([float(i[2]) for i in ps_lg_arr])
    ps_lg_pre_avg = np.mean([float(i[3]) for i in ps_lg_arr])
    ps_lg_rec_avg = np.mean([float(i[4]) for i in ps_lg_arr])
    ps_lg_f1__avg = np.mean([float(i[5]) for i in ps_lg_arr])

    ps_xg_arr = [i for i in p if i[1] == "psm_xgboost"]
    ps_xg_acc_avg = np.mean([float(i[2]) for i in ps_xg_arr])
    ps_xg_pre_avg = np.mean([float(i[3]) for i in ps_xg_arr])
    ps_xg_rec_avg = np.mean([float(i[4]) for i in ps_xg_arr])
    ps_xg_f1__avg = np.mean([float(i[5]) for i in ps_xg_arr])

    x = np.arange(2)
    x_acc = x - width * 1.5
    x_pre = x - width / 2
    x_rec = x + width / 2
    x_f1_ = x + width * 1.5

    fig, ax = plt.subplots(figsize=(10, 6))
    b_acc = ax.bar(
        x_acc,
        [ps_lg_acc_avg, ps_xg_acc_avg],
        width=width,
        label="accuracy",
        color="#f9766e",
        edgecolor="k",
        zorder=2,
    )
    b_pre = ax.bar(
        x_pre,
        [ps_lg_pre_avg, ps_xg_pre_avg],
        width=width,
        label="precision",
        color="#cccc11",
        edgecolor="k",
        zorder=2,
    )
    b_rec = ax.bar(
        x_rec,
        [ps_lg_rec_avg, ps_xg_rec_avg],
        width=width,
        label="recall",
        color="#66ccff",
        edgecolor="k",
        zorder=2,
    )
    b_f1_ = ax.bar(
        x_f1_,
        [ps_lg_f1__avg, ps_xg_f1__avg],
        width=width,
        label="f1",
        color="#39c5bb",
        edgecolor="k",
        zorder=2,
    )

    add_value_labels(ax, b_acc)
    add_value_labels(ax, b_pre)
    add_value_labels(ax, b_rec)
    add_value_labels(ax, b_f1_)

    ax.set_xticks(x)
    ax.set_xticklabels(["Logistic Regression", "XGBoost"])
    ax.legend()
    ax.set_title("Pooled Stock Logistic Regression vs XGBoost")
    fig.savefig("./outputs/all_models/ps_lg_xg.png")
    plt.close()

# ROC-AUC Single Stock LG vs XG
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    models_to_plot = ["logistic_regression", "xgboost"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
    for model, color in zip(models_to_plot, colors):
        all_true = []
        all_proba = []
        for symbol in comparison_df["symbol"].unique():
            if symbol in prediction_files and model in prediction_files[symbol]:
                p_y_true, y_proba = load_predictions(prediction_files[symbol][model])
                all_true.extend(p_y_true)
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
    ax.set_title("ROC Curves for Single Stock Logistic Regression vs XGBoost")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig("./outputs/all_models/ss_lg_xg_roc_auc.png")
    plt.close()

# ROC-AUC Pooled Stock LG vs XG
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    models_to_plot = ["psm_logistic_regression", "psm_xgboost"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
    for model, color in zip(models_to_plot, colors):
        all_true = []
        all_proba = []
        for symbol in comparison_df["symbol"].unique():
            if symbol in prediction_files and model in prediction_files[symbol]:
                p_y_true, y_proba = load_predictions(prediction_files[symbol][model])
                all_true.extend(p_y_true)
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
    ax.set_title("ROC Curves for Pooled Stock Logistic Regression vs XGBoost")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig("./outputs/all_models/ps_lg_xg_roc_auc.png")
    plt.close()

# Single Stock CNN Universal vs Specific
for _ in [0]:
    ss_cu_arr = [i for i in p if i[1] == "cnn_universal"]
    ss_cu_acc_avg = np.mean([float(i[2]) for i in ss_cu_arr])
    ss_cu_pre_avg = np.mean([float(i[3]) for i in ss_cu_arr])
    ss_cu_rec_avg = np.mean([float(i[4]) for i in ss_cu_arr])
    ss_cu_f1__avg = np.mean([float(i[5]) for i in ss_cu_arr])

    ss_cs_arr = [i for i in p if i[1] == "cnn_specific"]
    ss_cs_acc_avg = np.mean([float(i[2]) for i in ss_cs_arr])
    ss_cs_pre_avg = np.mean([float(i[3]) for i in ss_cs_arr])
    ss_cs_rec_avg = np.mean([float(i[4]) for i in ss_cs_arr])
    ss_cs_f1__avg = np.mean([float(i[5]) for i in ss_cs_arr])

    x = np.arange(2)
    x_acc = x - width * 1.5
    x_pre = x - width / 2
    x_rec = x + width / 2
    x_f1_ = x + width * 1.5

    fig, ax = plt.subplots(figsize=(10, 6))
    b_acc = ax.bar(
        x_acc,
        [ss_cu_acc_avg, ss_cs_acc_avg],
        width=width,
        label="accuracy",
        color="#f9766e",
        edgecolor="k",
        zorder=2,
    )
    b_pre = ax.bar(
        x_pre,
        [ss_cu_pre_avg, ss_cs_pre_avg],
        width=width,
        label="precision",
        color="#cccc11",
        edgecolor="k",
        zorder=2,
    )
    b_rec = ax.bar(
        x_rec,
        [ss_cu_rec_avg, ss_cs_rec_avg],
        width=width,
        label="recall",
        color="#66ccff",
        edgecolor="k",
        zorder=2,
    )
    b_f1_ = ax.bar(
        x_f1_,
        [ss_cu_f1__avg, ss_cs_f1__avg],
        width=width,
        label="f1",
        color="#39c5bb",
        edgecolor="k",
        zorder=2,
    )

    add_value_labels(ax, b_acc)
    add_value_labels(ax, b_pre)
    add_value_labels(ax, b_rec)
    add_value_labels(ax, b_f1_)

    ax.set_xticks(x)
    ax.set_xticklabels(["CNN Universal", "CNN Specific"])
    ax.legend()
    ax.set_title("Single Stock CNN Universal vs CNN Specific")
    fig.savefig("./outputs/all_models/ss_cu_cs.png")
    plt.close()

# ROC-AUC CNN Universal
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    models_to_plot = ["cnn_universal"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
    for model, color in zip(models_to_plot, colors):
        all_true = []
        all_proba = []
        for symbol in comparison_df["symbol"].unique():
            if symbol in prediction_files and model in prediction_files[symbol]:
                p_y_true, y_proba = load_predictions(prediction_files[symbol][model])
                all_true.extend(p_y_true)
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
    ax.set_title("ROC Curves for CNN Universal")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig("./outputs/all_models/ss_cu_roc_auc.png")
    plt.close()

# ROC-AUC CNN Specific
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    models_to_plot = ["cnn_specific"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_to_plot)))
    for model, color in zip(models_to_plot, colors):
        all_true = []
        all_proba = []
        for symbol in comparison_df["symbol"].unique():
            if symbol in prediction_files and model in prediction_files[symbol]:
                p_y_true, y_proba = load_predictions(prediction_files[symbol][model])
                all_true.extend(p_y_true)
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
    ax.set_title("ROC Curves for CNN Specific")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig("./outputs/all_models/ss_cs_roc_auc.png")
    plt.close()

# Combined Metrics Heatmap
for _ in [0]:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    metrics = ["accuracy", "precision", "recall", "f1"]
    titles = [
        "Accuracy Heatmap",
        "Precision Heatmap",
        "Recall Heatmap",
        "F1 Score Heatmap",
    ]

    all_metrics_data = comparison_df[metrics].values
    global_min = np.min(all_metrics_data)
    global_max = np.max(all_metrics_data)

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        sns.heatmap(
            comparison_df.pivot_table(
                index="symbol", columns="model", values=metric, aggfunc=np.mean
            ),
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            vmin=global_min,
            vmax=global_max,
            cbar_kws={"label": metric.capitalize()},
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig("./outputs/all_models/combined_metrics_heatmap.png")
    plt.close()


# Accuracy Heatmap Single Stock LG vs XG vs CNN Universal
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        comparison_df[
            comparison_df["model"].isin(
                ["logistic_regression", "xgboost", "cnn_universal"]
            )
        ].pivot_table(
            index="symbol", columns="model", values="accuracy", aggfunc=np.mean
        ),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_title("Accuracy Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig("./outputs/all_models/ss_xg_lg_cu_acc_hm.png")
    plt.close()

# Precision Heatmap Single Stock LG vs XG vs CNN Universal
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        comparison_df[
            comparison_df["model"].isin(
                ["logistic_regression", "xgboost", "cnn_universal"]
            )
        ].pivot_table(
            index="symbol", columns="model", values="precision", aggfunc=np.mean
        ),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Precision"},
        ax=ax,
    )
    ax.set_title("Precision Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig("./outputs/all_models/ss_xg_lg_cu_pre_hm.png")
    plt.close()

# Recall Heatmap Single Stock LG vs XG vs CNN Universal
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        comparison_df[
            comparison_df["model"].isin(
                ["logistic_regression", "xgboost", "cnn_universal"]
            )
        ].pivot_table(
            index="symbol", columns="model", values="recall", aggfunc=np.mean
        ),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Recall"},
        ax=ax,
    )
    ax.set_title("Recall Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig("./outputs/all_models/ss_xg_lg_cu_rec_hm.png")
    plt.close()

# F1 Score Heatmap Single Stock LG vs XG vs CNN Universal
for _ in [0]:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        comparison_df[
            comparison_df["model"].isin(
                ["logistic_regression", "xgboost", "cnn_universal"]
            )
        ].pivot_table(index="symbol", columns="model", values="f1", aggfunc=np.mean),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "F1 Score"},
        ax=ax,
    )
    ax.set_title("F1 Score Heatmap")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.savefig("./outputs/all_models/ss_xg_lg_cu_f1__hm.png")
    plt.close()

# Colored Scatter Single Stock CNN Universal GOOGL
for _ in [0]:
    symbol = "GOOGL"
    model_name = "cnn_universal"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(p_dates, p_y_pred_proba, c=p_y_pred, cmap="coolwarm")
    ax.set_title(f"{symbol} CNN Universal Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction Probability")
    ax.grid(alpha=0.3)
    fig.savefig(f"./outputs/all_models/ss_cu_scatter_{symbol}.png")
    plt.close()

# Colored Scatter Single Stock CNN Specific GOOGL
for _ in [0]:
    symbol = "GOOGL"
    model_name = "cnn_specific"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(p_dates, p_y_pred_proba, c=p_y_pred, cmap="coolwarm")
    ax.set_title(f"{symbol} CNN Specific Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction Probability")
    ax.grid(alpha=0.3)
    fig.savefig(f"./outputs/all_models/ps_cu_scatter_{symbol}.png")
    plt.close()

# Colored Scatter Single Stock XG BABA
for _ in [0]:
    symbol = "BABA"
    model_name = "xgboost"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_dates = pd.to_datetime(p_dates)
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]
    c = (p_y_pred == p_y_true).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        p_dates,
        p_y_true,
        c=c,
        cmap="RdYlGn",
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    for i, (d, p, a) in enumerate(zip(p_dates, p_y_pred, p_y_true)):
        if p == a:
            marker = "o"
            color = "green"
        else:
            marker = "x"
            color = "red"
        ax.scatter(
            d,
            a,
            marker=marker,
            s=50,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_title(f"{symbol} XGBoost Specific Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction Probability")
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    fig.savefig(f"./outputs/all_models/ss_xg_scatter_{symbol}.png")
    plt.close()

# Cumulative Accuracy Curve Single Stock XG BABA
for _ in [0]:
    symbol = "BABA"
    model_name = "xgboost"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_dates = pd.to_datetime(p_dates)
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]
    c = (p_y_pred == p_y_true).astype(int)

    cumulative_accuracy = c.cumsum() / np.arange(1, len(c) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_dates, cumulative_accuracy, linewidth=2, color="blue", label="Cumulative Accuracy")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Random Guess (0.5)")
    ax.axhline(
        y=c.mean(),
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean Accuracy: {c.mean():.2%}",
    )

    ax.fill_between(
        p_dates,
        0.5,
        cumulative_accuracy,
        where=(cumulative_accuracy >= 0.5),
        color="green",
        alpha=0.4,
        label="Better than Random",
    )
    ax.fill_between(
        p_dates,
        0.5,
        cumulative_accuracy,
        where=(cumulative_accuracy < 0.5),
        color="red",
        alpha=0.4,
        label="Worse than Random",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax.set_title(f"{symbol} XGBoost Cumulative Prediction Accuracy Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.tight_layout()
    fig.savefig(f"./outputs/all_models/ss_xg_acc_curve_{symbol}.png")
    plt.close()

# Colored Scatter Single Stock XG NVDA
for _ in [0]:
    symbol = "NVDA"
    model_name = "xgboost"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_dates = pd.to_datetime(p_dates)
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]
    c = (p_y_pred == p_y_true).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        p_dates,
        p_y_true,
        c=c,
        cmap="RdYlGn",
        s=80,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    for i, (d, p, a) in enumerate(zip(p_dates, p_y_pred, p_y_true)):
        if p == a:
            marker = "o"
            color = "green"
        else:
            marker = "x"
            color = "red"
        ax.scatter(
            d,
            a,
            marker=marker,
            s=50,
            color=color,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_title(f"{symbol} XGBoost Specific Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prediction Probability")
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    fig.savefig(f"./outputs/all_models/ss_xg_scatter_{symbol}.png")
    plt.close()

# Cumulative Accuracy Curve Single Stock XG NVDA
for _ in [0]:
    symbol = "NVDA"
    model_name = "xgboost"
    prediction_file = f"./outputs/all_models/future_predictions/{symbol}_{model_name}_future_predictions.npz"

    p = np.load(prediction_file, allow_pickle=True)

    p_dates = p["dates"]
    p_dates = pd.to_datetime(p_dates)
    p_y_true = p["y_true"]
    p_y_pred = p["y_pred"]
    p_y_pred_proba = p["y_pred_proba"]
    c = (p_y_pred == p_y_true).astype(int)

    cumulative_accuracy = c.cumsum() / np.arange(1, len(c) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_dates, cumulative_accuracy, linewidth=2, color="blue", label="Cumulative Accuracy")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5, label="Random Guess (0.5)")
    ax.axhline(
        y=c.mean(),
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean Accuracy: {c.mean():.2%}",
    )

    ax.fill_between(
        p_dates,
        0.5,
        cumulative_accuracy,
        where=(cumulative_accuracy >= 0.5),
        color="green",
        alpha=0.4,
        label="Better than Random",
    )
    ax.fill_between(
        p_dates,
        0.5,
        cumulative_accuracy,
        where=(cumulative_accuracy < 0.5),
        color="red",
        alpha=0.4,
        label="Worse than Random",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax.set_title(f"{symbol} XGBoost Cumulative Prediction Accuracy Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.tight_layout()
    fig.savefig(f"./outputs/all_models/ss_xg_acc_curve_{symbol}.png")
    plt.show()
    plt.close()