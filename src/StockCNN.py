import torch.nn as nn


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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
