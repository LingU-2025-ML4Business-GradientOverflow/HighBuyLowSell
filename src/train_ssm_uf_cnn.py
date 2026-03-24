import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from feature_pipeline_universal import feature_pipeline
from StockCNN import StockCNN


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CNN model training for stock prediction"
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
        default="outputs/ssm_uf_cnn",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
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
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Print training progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    # Load best model
    model.load_state_dict(best_model_state)

    return model, time_steps


def save_model(model: nn.Module, symbol: str, output_dir: str) -> None:
    """
    Save trained model to disk

    Args:
        model: Trained model
        symbol: Stock symbol
        output_dir: Output directory
    """
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{symbol}_ssm_uf_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def run_single_stock_experiment(
    data_path: str,
    symbol: str,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: str = "outputs/ssm_uf_cnn",
) -> Dict:
    """
    Run single stock experiment using feature pipeline

    Args:
        data_path: Path to data file
        symbol: Stock symbol
        test_size: Test set ratio
        random_state: Random seed
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Output directory for models

    Returns:
        Dict: dictionary containing experiment results
    """
    print(f"Starting single stock experiment: {symbol}")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        data_path, symbol, test_size, random_state
    )

    # Get feature column names
    feature_cols = X_train.columns.tolist()

    # Train model
    cnn_model, time_steps = train_cnn(
        X_train, y_train, random_state, epochs, batch_size, learning_rate
    )

    # Save model
    save_model(cnn_model, symbol, output_dir)

    # Return results
    return {
        "symbol": symbol,
        "feature_cols": feature_cols,
        "time_steps": time_steps,
    }


def main() -> None:
    """Main function"""
    args = parse_args()

    # Run experiments for all stocks
    all_results = []
    for symbol in args.symbols:
        result = run_single_stock_experiment(
            args.data_path,
            symbol,
            args.test_size,
            args.random_state,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.output_dir,
        )
        all_results.append(result)

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_path / "training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Training results saved to {output_path}")


if __name__ == "__main__":
    main()
