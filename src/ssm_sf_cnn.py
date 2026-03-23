import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
        default="outputs/ssm_sf_cnn",
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
        Initialize CNN model for stock prediction

        Args:
            input_size: Number of input features
            time_steps: Time steps
        """
        super(StockCNN, self).__init__()
        self.time_steps = time_steps

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)
        # Modify pooling layer, use kernel_size=1 or remove second pooling layer
        self.pool2 = nn.MaxPool1d(kernel_size=1)  # Modified to kernel_size=1

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 50)
        self.fc2 = nn.Linear(50, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass function

        Args:
            x: Input tensor with shape (batch_size, time_steps, features)

        Returns:
            Output tensor with shape (batch_size, 1)
        """
        # Input shape: (batch_size, time_steps, features)
        # Convert to (batch_size, features, time_steps) for Conv1d
        x = x.permute(0, 2, 1)

        # First convolution and pooling
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)

        # Second convolution and pooling
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)

        # Flatten and connect to fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))

        return x


def prepare_sequences(
    X: pd.DataFrame, y: pd.Series, time_steps: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare time series data

    Args:
        X: Feature data
        y: Target data
        time_steps: Time steps

    Returns:
        Tuple: (X_seq, y_seq) sequenced data and labels
    """
    X_seq = []
    y_seq = []

    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i : (i + time_steps)].values)
        y_seq.append(y.iloc[i + time_steps])

    return np.array(X_seq), np.array(y_seq)



def train_cnn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
) -> Tuple[nn.Module, int]:
    """
    Train CNN model using PyTorch

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate

    Returns:
        Tuple: (model, time_steps) Trained model and time steps
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Define time steps
    time_steps = 5

    # Prepare sequence data
    X_seq, y_seq = prepare_sequences(X_train, y_train, time_steps)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_seq)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)  # Add dimension to match output

    # Create data loaders
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    input_size = X_train.shape[1]
    model = StockCNN(input_size=input_size, time_steps=time_steps)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        # Early stopping mechanism
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Print training progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Load best model
    model.load_state_dict(best_model_state)

    return model, time_steps


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, time_steps: int = None
) -> Dict[str, float]:
    """
    Evaluate model performance

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        time_steps: Time steps(CNN only)

    Returns:
        Dict: dictionary containing various evaluation metrics
    """
    # Initialize predictions and probabilities
    y_pred = None
    y_pred_prob = None

    # For CNN model, need to create sequence data
    if time_steps is not None:
        X_seq, y_seq = prepare_sequences(X_test, y_test, time_steps)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq)

        # 预测
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_tensor).numpy().flatten()

        # Convert to binary predictions
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Use sequence labels
        y_test = y_seq
    else:
        # For non-CNN models, use original prediction method
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probability for positive class

    # Calculate evaluation metrics
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
            # Handle possible calculation errors
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
    Get feature importance for CNN model

    Args:
        model: Trained CNN model
        feature_cols: List of feature column names

    Returns:
        Dict: mapping of feature names to importance scores
    """
    # Get first convolutional layer weights
    conv1_weights = (
        model.conv1.weight.data.numpy()
    )  # shape: (out_channels, in_channels, kernel_size)

    # Calculate mean absolute weight for each input feature channel
    feature_importance = np.mean(
        np.abs(conv1_weights), axis=(0, 2)
    )  # shape: (in_channels,)

    # Normalize to 0-1 range
    feature_importance = (feature_importance - feature_importance.min()) / (
        feature_importance.max() - feature_importance.min()
    )

    # Map feature names to importance scores
    return dict(zip(feature_cols, feature_importance))


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
        Dict: dictionary containing experiment results
    """
    logger.info(f"Starting single stock experiment: {symbol}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        data_path, symbol, test_size, random_state
    )

    # Get feature column names
    feature_cols = X_train.columns.tolist()

    # Train model
    cnn_model, time_steps = train_cnn(X_train, y_train, random_state)

    # Evaluate model
    cnn_results = evaluate_model(cnn_model, X_test, y_test, time_steps)

    # Return results
    return {
        "symbol": symbol,
        "cnn": cnn_results,
        "feature_importance": {
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
            if model_name in ["cnn"]:
                row = {"symbol": symbol, "model": model_name}
                row |= metrics
                comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Simple visualization
    plt.figure(figsize=(15, 10))

    # Subplot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    pivot_acc = comparison_df.pivot(index="symbol", columns="model", values="accuracy")
    pivot_acc.plot(kind="bar", ax=plt.gca())
    plt.title("Accuracy Comparison by Stock and Model")
    plt.xlabel("stock symbol")
    plt.ylabel("Accuracy")
    plt.legend(title="Model")
    plt.xticks(rotation=45)

    # Subplot 2: Average model performance (including ROC-AUC)
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

    # Subplot 3: ROC-AUC heatmap
    plt.subplot(2, 2, 3)
    pivot_roc = comparison_df.pivot(index="symbol", columns="model", values="roc_auc")
    sns.heatmap(
        pivot_roc, annot=True, fmt=".3f", cmap="YlOrRd", cbar_kws={"label": "ROC-AUC"}
    )
    plt.title("ROC-AUC Heatmap")

    # Subplot 4: Boxplot (model stability)
    plt.subplot(2, 2, 4)
    comparison_df.boxplot(column="roc_auc", by="model", ax=plt.gca())
    plt.title("Stability of Models (ROC-AUC Distribution)")
    plt.suptitle("")
    plt.xlabel("Model")
    plt.ylabel("ROC-AUC")

    plt.tight_layout()
    plt.show()

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


def save_results(results: List[Dict], output_dir: str) -> pd.DataFrame:
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
    return comparison_df


def generate_conclusions(comparison_df: List[Dict]) -> pd.DataFrame:
    """Generate experiment conclusions"""

    # Find best performing stock and model
    best_accuracy = comparison_df.loc[comparison_df["accuracy"].idxmax()]

    # Compare model stability
    model_stability = comparison_df.groupby("model")["accuracy"].std().to_dict()

    # Compare stock modelability
    stock_modelability = comparison_df.groupby("symbol")["accuracy"].mean().to_dict()

    # Determine recommended model (based on stability and performance)
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
            "worth_retaining": True,
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
    comparison_df = save_results(all_results, args.output_dir)

    # Generate conclusions
    conclusions = generate_conclusions(comparison_df)

    # Print conclusions
    logger.info("Experiment Conclusions:")
    logger.info(f"Best Performance: {conclusions['best_performing']}")
    logger.info(f"Model Stability: {conclusions['model_stability']}")
    logger.info(f"Stock Modelability: {conclusions['stock_modelability']}")
    logger.info(f"Recommendations: {conclusions['recommendations']}")


if __name__ == "__main__":
    main()