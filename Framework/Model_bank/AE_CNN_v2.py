from .building_blocks.custom_blocks import CNN_Block_1D, CNN_Block_inv_1D
import torch
import torch.nn as nn


class CNNAEV2(nn.Module):
    def __init__(self, no_channels=1):
        super(CNNAEV2, self).__init__()
        self.channels = no_channels
        self.model_name = "CNN_AE_v2"

        # Encoder: 1D Convolution to reduce the sequence dimension
        self.encoder = nn.Sequential(
            CNN_Block_1D(self.channels, 16),
            nn.Dropout(0.5),
            CNN_Block_1D(16, 32),
            nn.Dropout(0.2),
            CNN_Block_1D(32, 64),

        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            CNN_Block_inv_1D(64, 32),
            nn.Dropout(0.2),
            CNN_Block_inv_1D(32, 16),
            nn.Dropout(0.1),
            CNN_Block_inv_1D(16, self.channels),
        )

    def forward(self, x):
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        return x