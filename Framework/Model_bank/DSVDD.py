import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    def __init__(self):
        super(DeepSVDD, self).__init__()
        self.features = nn.Sequential(
             nn.Conv1d(1, 8, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool1d(2),
             nn.Conv1d(8, 16, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool1d(2),
             nn.Conv2d(16, 32, kernel_size=3, padding=1),
             nn.ReLU(),
             nn.MaxPool1d(2),
            )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(288, 32)  # We will dynamically set this

    def forward(self, x):
        x = self.features(x)
        x = self.fc(self.flatten(x))

        return x