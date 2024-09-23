# Description: This file contains the model definitions for the autoencoder models.
from torch import nn


class AE1DCNN(nn.Module):
    """
    Autoencoder model with 1D CNN layers.
    """
    def __init__(self, ):
        super(AE1DCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AEFC(nn.Module):
    """
    Autoencoder model with fully connected layers.
    """
    def __init__(self, ):
        super(AEFC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(72, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 24),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 72),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
