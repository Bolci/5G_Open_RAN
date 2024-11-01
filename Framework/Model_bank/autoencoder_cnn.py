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


class CNNAutoencoderV3(nn.Module):
    def __init__(self,
                 no_channels=1,
                 num_layers=3,
                 init_channels=16,
                 growth_factor=2,
                 kernel_size=3,
                 pool_type='max',
                 dropout=0.1):
        """
        :param no_channels: Number of input channels.
        :param num_layers: Number of layers in encoder and decoder.
        :param init_channels: Number of output channels for the first encoder layer.
        :param growth_factor: Factor to multiply channels with each layer.
        :param kernel_size: Kernel size for convolutional layers.
        :param pool_type: Pooling type, either 'max' or 'avg'.
        :param dropout: Dropout rate to apply between layers.
        """
        super(CNNAutoencoderV3, self).__init__()

        self.channels = no_channels
        self.model_name = "CNN_AE_Custom"
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Build Encoder with dynamic configuration
        in_channels = self.channels
        for i in range(num_layers):
            out_channels = init_channels * (growth_factor ** i)
            self.encoder.add_module(f"encoder_conv_{i}",
                                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
            self.encoder.add_module(f"encoder_dropout_{i}", nn.Dropout(dropout))
            self.encoder.add_module(f"encoder_relu_{i}", nn.ReLU(True))
            if pool_type == 'max':
                self.encoder.add_module(f"encoder_pool_{i}", nn.MaxPool1d(2))
            elif pool_type == 'avg':
                self.encoder.add_module(f"encoder_pool_{i}", nn.AvgPool1d(2))
            in_channels = out_channels

        # Build Decoder with upsampling
        for i in range(num_layers - 1, -1, -1):
            out_channels = init_channels * (growth_factor ** i)
            self.decoder.add_module(f"decoder_upsample_{i}", nn.Upsample(scale_factor=2))
            self.decoder.add_module(f"decoder_convtrans_{i}", nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                                                                 padding=kernel_size // 2))
            self.decoder.add_module(f"decoder_dropout_{i}", nn.Dropout(dropout))
            self.decoder.add_module(f"decoder_relu_{i}", nn.ReLU(True))
            in_channels = out_channels

        # Final output layer to match the original input channels
        self.decoder.add_module("decoder_output",
                                nn.ConvTranspose1d(in_channels, self.channels, kernel_size, padding=kernel_size // 2))
        self.decoder.add_module("decoder_sigmoid", nn.Sigmoid())

    def forward(self, x):
        # Pass through encoder
        x = self.encoder(x)
        # Pass through decoder
        x = self.decoder(x)
        return x
