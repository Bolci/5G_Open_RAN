import torch
import torch.nn as nn
from numpy.ma.core import append


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_channels: int = 2):
        super(LSTMAutoencoder, self).__init__()
        self.model_name = "LSTM_AE"

        self.encoder_lstm = nn.LSTM(input_size=input_channels, hidden_size=64, num_layers=1, batch_first=True)

        self.latent_lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)

        self.decoder_lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)

        self.output_layer = nn.Linear(64, input_channels)

    def forward(self, x):
        encoded, _ = self.encoder_lstm(x)
        latent, _ = self.latent_lstm(encoded)
        decoded, _ = self.decoder_lstm(latent)
        output = self.output_layer(decoded)

        return output


class LSTMAutoencoderCustom(nn.Module):
    def __init__(self,
                 input_dimensions: int = 72,
                 expansion_dim: int = 2,
                 no_layers_per_module: int = 2,
                 num_layers_per_layer: int = 1,
                 init_channels: int = 16,
                 dropout: float = 0.1,
                 device = 'cuda'):
        super(LSTMAutoencoderCustom, self).__init__()
        self.model_name = "LSTM_AE_Custom"
        self.encoder = []
        self.decoder = []

        in_size = input_dimensions
        for x in range(no_layers_per_module):
            output_size = (expansion_dim ** x) * init_channels
            self.encoder.append(
                                    nn.LSTM(input_size=in_size,
                                            hidden_size=output_size,
                                            num_layers=num_layers_per_layer,
                                            batch_first=True).to(device))


            in_size = output_size
        self.dropout = nn.Dropout(dropout).to(device)
        self.latent_lstm = nn.LSTM(input_size=in_size, hidden_size=in_size, num_layers=num_layers_per_layer, batch_first=True).to(device)

        for x in range(no_layers_per_module-1, -1, -1):
            output_size = (expansion_dim ** x) * init_channels
            self.decoder.append(
                                    nn.LSTM(input_size=in_size,
                                            hidden_size=output_size,
                                            num_layers=num_layers_per_layer,
                                            batch_first=True).to(device))
            #self.decoder.append(nn.Dropout(dropout).to(device))

            in_size = output_size

        self.output_layer = nn.Linear(in_size, input_dimensions).to(device)

    def forward(self, x):

        for layer in self.encoder:
            x, _ = layer(x)
            x = self.dropout(x)

        x, _ = self.latent_lstm(x)

        for layer in self.decoder:
            x, _ = layer(x)
            x = self.dropout(x)

        output = self.output_layer(x)
        return output