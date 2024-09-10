import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder: 1D Convolution to reduce the sequence dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=2, padding=1),  # (2, 144) -> (16, 72)
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # (16, 72) -> (32, 36)
            nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, 36) -> (64, 18)
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)  # (64, 18) -> (128, 9)
        )
        
        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 9) -> (64, 18)
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),   # (64, 18) -> (32, 36)
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),   # (32, 36) -> (16, 72)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),    # (16, 72) -> (2, 144)
        )
    
    def forward(self, x):
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x
