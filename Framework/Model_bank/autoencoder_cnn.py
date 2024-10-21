import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self, no_channels=1, dropout=0.1):
        super(CNNAutoencoder, self).__init__()
        self.channels = no_channels
        self.model_name = "CNN_AE"

        # Encoder: 1D Convolution to reduce the sequence dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (32, 36)
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (64, 18)
            nn.ReLU(True),
        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (32, 36)
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (16, 72)
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, self.channels, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (2, 144)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNNAutoencoderV2(nn.Module):
    def __init__(self, no_channels=1, dropout=0.1):
        super(CNNAutoencoderV2, self).__init__()
        self.channels = no_channels
        self.model_name = "CNN_AE"

        # Encoder: 1D Convolution to reduce the sequence dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout),
            nn.ELU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (32, 36)
            nn.Dropout(dropout),
            nn.ELU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (64, 18)
            nn.Dropout(dropout),
            nn.ELU(True),
        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (32, 36)
            nn.Dropout(dropout),
            nn.ELU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (16, 72)
            nn.Dropout(dropout),
            nn.ELU(True),
            nn.ConvTranspose1d(16, self.channels, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (2, 144)
            nn.ELU()
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x