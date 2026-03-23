import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from feature_pipeline import feature_pipeline


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        default="outputs/issue3a",
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

    X_train.replace((np.inf, -np.inf), 0, inplace=True)

    return X_train, X_test, y_train, y_test


class StockCNN(nn.Module):
    def __init__(self, input_size: int, time_steps: int = 5):
        """
        初始化CNN模型用于股票预测

        Args:
            input_size: 输入特征数量
            time_steps: 时间步长
        """
        super(StockCNN, self).__init__()
        self.time_steps = time_steps

        # 卷积层
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        # 修改池化层，使用kernel_size=1或移除第二个池化层
        self.pool2 = nn.MaxPool1d(kernel_size=1)  # 修改为kernel_size=1

        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 50)
        self.fc2 = nn.Linear(50, 1)

        # Dropout层
        self.dropout = nn.Dropout(p=0.3)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入张量，形状为 (batch_size, time_steps, features)

        Returns:
            输出张量，形状为 (batch_size, 1)
        """
        # 输入形状: (batch_size, time_steps, features)
        # 转换为 (batch_size, features, time_steps) 适配Conv1d
        x = x.permute(0, 2, 1)

        # 第一层卷积和池化
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)

        # 第二层卷积和池化
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        # 展平并连接到全连接层
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x


def prepare_sequences(
    X: pd.DataFrame, y: pd.Series, time_steps: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    准备时间序列数据

    Args:
        X: 特征数据
        y: 目标数据
        time_steps: 时间步长

    Returns:
        Tuple: (X_seq, y_seq) 序列化后的数据和标签
    """
    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i : (i + time_steps)].values)
        y_seq.append(y.iloc[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> LogisticRegression:
    """Train logistic regression model"""
    model = LogisticRegression(
        random_state=random_state, max_iter=5000, solver="saga", tol=1e-3
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> xgb.XGBClassifier:
    """Train XGBoost model with optimized parameters"""
    model = xgb.XGBClassifier(
        random_state=random_state,
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


def train_cnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Tuple[nn.Module, int]:
    """
    使用PyTorch训练CNN模型

    Args:
        X_train: 训练特征
        y_train: 训练目标
        random_state: 随机种子
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率

    Returns:
        Tuple: (model, time_steps) 训练好的模型和时间步长
    """
    # 设置随机种子以确保可重复性
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # 定义时间步长
    time_steps = 5

    # 准备序列数据
    X_seq, y_seq = prepare_sequences(X_train, y_train, time_steps)

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)  # 添加维度以匹配输出

    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    input_size = X_train.shape[1]
    model = StockCNN(input_size=input_size, time_steps=time_steps)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, time_steps


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, time_steps: int = None
) -> Dict[str, float]:
    """
    评估模型性能

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试目标
        time_steps: 时间步长（仅CNN需要）

    Returns:
        Dict: 包含各种评估指标的字典
    """
    # 初始化预测结果和概率
    y_pred = None
    y_pred_prob = None

    # 对于CNN模型，需要创建序列数据
    if time_steps is not None:
        X_seq, y_seq = prepare_sequences(X_test, y_test, time_steps)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_seq)

        # 预测
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_tensor).numpy().flatten()

        # 转换为二分类预测
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 使用序列化的标签
        y_test = y_seq
    else:
        # 对于非CNN模型，使用原始预测方法
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

    # 计算评估指标
    results = {}
    for metric_name, metric_func in METRICS.items():
        try:
            if metric_name == "roc_auc":
                # ROC-AUC需要概率值
                results[metric_name] = metric_func(y_test, y_pred_prob)
            else:
                # 其他指标使用二分类预测
                results[metric_name] = metric_func(y_test, y_pred)
        except Exception as e:
            # 处理可能的计算错误
            logger.warning(f"Error calculating {metric_name}: {str(e)}")
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


def get_cnn_feature_importance(
    model: nn.Module, feature_cols: List[str]
) -> Dict[str, float]:
    """
    获取CNN模型的特征重要性

    Args:
        model: 训练好的CNN模型
        feature_cols: 特征列名列表

    Returns:
        Dict: 特征名到重要性分数的映射
    """
    # 获取第一层卷积层的权重
    conv1_weights = (
        model.conv1.weight.data.numpy()
    )  # shape: (out_channels, in_channels, kernel_size)

    # 计算每个输入特征通道的权重绝对值均值
    feature_importance = np.mean(
        np.abs(conv1_weights), axis=(0, 2)
    )  # shape: (in_channels,)

    # 归一化到0-1范围
    feature_importance = (feature_importance - feature_importance.min()) / (
        feature_importance.max() - feature_importance.min()
    )

    # 创建特征名到重要性分数的映射
    importance_dict = dict(zip(feature_cols, feature_importance))

    return importance_dict


def run_single_stock_experiment(
    data_path: str, symbol: str, test_size: float = 0.2, random_state: int = 42
) -> Dict:
    """
    使用特征管道运行单股实验

    Args:
        data_path: 数据文件路径
        symbol: 股票代码
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        Dict: 包含实验结果的字典
    """
    logger.info(f"Starting single stock experiment: {symbol}")

    # 准备数据
    X_train, X_test, y_train, y_test = prepare_data(
        data_path, symbol, test_size, random_state
    )

    # 获取特征列名
    feature_cols = X_train.columns.tolist()

    # 训练模型
    lr_model = train_logistic_regression(X_train, y_train, random_state)
    xgb_model = train_xgboost(X_train, y_train, random_state)
    cnn_model, time_steps = train_cnn(X_train, y_train, random_state)

    # 评估模型
    lr_results = evaluate_model(lr_model, X_test, y_test)
    xgb_results = evaluate_model(xgb_model, X_test, y_test)
    cnn_results = evaluate_model(cnn_model, X_test, y_test, time_steps)

    # 返回结果
    return {
        "symbol": symbol,
        "logistic_regression": lr_results,
        "xgboost": xgb_results,
        "cnn": cnn_results,
        "feature_importance": {
            "logistic_regression": dict(zip(feature_cols, lr_model.coef_[0])),
            "xgboost": dict(zip(feature_cols, xgb_model.feature_importances_)),
            "cnn": get_cnn_feature_importance(cnn_model, feature_cols),
        },
        "time_steps": time_steps,
    }


def compare_models(results: List[Dict]) -> pd.DataFrame:
    """比较不同股票和模型的结果，并生成简单可视化"""
    comparison_data = []

    for result in results:
        symbol = result["symbol"]
        for model_name, metrics in result.items():
            if model_name in ["logistic_regression", "xgboost", "cnn"]:
                row = {"symbol": symbol, "model": model_name}
                row.update(metrics)
                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # 简单可视化
    plt.figure(figsize=(15, 10))

    # 子图1：准确率比较
    plt.subplot(2, 2, 1)
    pivot_acc = comparison_df.pivot(index="symbol", columns="model", values="accuracy")
    pivot_acc.plot(kind="bar", ax=plt.gca())
    plt.title("Accuracy Comparison by Stock and Model")
    plt.xlabel("stock symbol")
    plt.ylabel("Accuracy")
    plt.legend(title="Model")
    plt.xticks(rotation=45)

    # 子图2：模型平均性能（包含ROC-AUC）
    plt.subplot(2, 2, 2)
    model_avg = comparison_df.groupby("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean()
    model_avg.plot(kind="bar", ax=plt.gca())
    plt.title("Average Performance by Model")
    plt.xlabel("Model")
    plt.ylabel("Average Score")
    plt.legend(title="Metrics")
    plt.xticks(rotation=0)

    # 子图3：ROC-AUC热力图
    plt.subplot(2, 2, 3)
    pivot_roc = comparison_df.pivot(index="symbol", columns="model", values="roc_auc")
    sns.heatmap(
        pivot_roc, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "ROC-AUC"}
    )
    plt.title("ROC-AUC Heatmap")

    # 子图4：箱线图（模型稳定性）
    plt.subplot(2, 2, 4)
    comparison_df.boxplot(column="roc_auc", by="model", ax=plt.gca())
    plt.title("Stability of Models (ROC-AUC Distribution)")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel("ROC-AUC")

    plt.tight_layout()
    plt.show()

    # 打印关键信息到控制台
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

    logger.info(f"Results saved to {output_path}")


def generate_conclusions(results: List[Dict]) -> Dict:
    """生成实验结论"""
    comparison_df = compare_models(results)

    # 找到表现最好的股票和模型
    best_accuracy = comparison_df.loc[comparison_df["accuracy"].idxmax()]

    # 比较模型稳定性
    model_stability = comparison_df.groupby("model")["accuracy"].std().to_dict()

    # 比较股票可建模性
    stock_modelability = comparison_df.groupby("symbol")["accuracy"].mean().to_dict()

    # 确定推荐模型（基于稳定性和性能）
    models = list(model_stability.keys())
    recommended_model = min(
        models,
        key=lambda m: (
            model_stability[m],
            -comparison_df[comparison_df["model"] == m]["accuracy"].mean(),
        ),
    )

    return {
        "best_performing": {
            "symbol": best_accuracy["symbol"],
            "model": best_accuracy["model"],
            "accuracy": best_accuracy["accuracy"],
        },
        "model_stability": model_stability,
        "stock_modelability": stock_modelability,
        "recommendations": {
            "worth_retaining": True,  # 假设CNN值得尝试
            "recommended_symbol": max(stock_modelability, key=stock_modelability.get),
            "recommended_model": recommended_model,
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
    logger.info("Experiment Conclusions:")
    logger.info(f"Best Performance: {conclusions['best_performing']}")
    logger.info(f"Model Stability: {conclusions['model_stability']}")
    logger.info(f"Stock Modelability: {conclusions['stock_modelability']}")
    logger.info(f"Recommendations: {conclusions['recommendations']}")


if __name__ == "__main__":
    main()
