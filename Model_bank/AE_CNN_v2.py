from .building_blocks.custom_blocks import CNN_Block_1D, CNN_Block_inv_1D
import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    def __init__(self, no_channels=1):
        super(CNNAutoencoder, self).__init__()
        self.channels = no_channels
        self.model_name = "CNN_AE_v2"

        # Encoder: 1D Convolution to reduce the sequence dimension
        self.encoder = nn.Sequential(
            CNN_Block_1D(self.channels, 16),
            CNN_Block_1D(16, 32),
            CNN_Block_1D(32, 64),

        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            CNN_Block_inv_1D(64, 32),
            CNN_Block_inv_1D(32, 16),
            CNN_Block_inv_1D(16, self.channels),
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x