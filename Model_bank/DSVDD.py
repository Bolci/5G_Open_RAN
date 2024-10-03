import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    def __init__(self):
        super(DeepSVDD, self).__init__()
        self.features = nn.Sequential(
             nn.Conv2d(1, 8, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(8, 16, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool2d(2),
             nn.Conv2d(16, 32, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool2d(2),
            )
        self.flatten = nn.Flatten(start_dim=1)
        # Initialize with a placeholder for in_features
        self.fc = nn.Linear(0, 32)  # We will dynamically set this

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        if self.fc.in_features == 0:  # Check if in_features is initialized
            self.fc.in_features = x.shape[1]  # Set the correct in_features
            self.fc.weight = nn.Parameter(torch.randn(32, x.shape[1]))  # Reinitialize weight
            self.fc.bias = nn.Parameter(torch.zeros(32))  # Reinitialize bias
        x = self.fc(x)
        return x