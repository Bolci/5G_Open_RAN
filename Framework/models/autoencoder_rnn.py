import torch.nn as nn


class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(RNNAutoencoder, self).__init__()
        self.name = "rnn_ae"
        # Encoder
        self.encoder = nn.Sequential()

        in_dim = input_dim
        for i, size in enumerate(hidden_dims):
            self.encoder.add_module(f"lstm_{i}", nn.LSTM(in_dim, size, batch_first=True))
            in_dim = size

        # Decoder
        self.decoder = nn.Sequential()
        for i, size in enumerate(reversed(hidden_dims)):
            self.decoder.add_module(f"lstm_{i}", nn.LSTM(in_dim, size, batch_first=True))
            in_dim = size
        self.decoder.add_module("output", nn.LSTM(in_dim, input_dim, batch_first=True))


    def forward(self, x):
        # Encoding the input sequence
        for layer in self.encoder:
            x, _ = layer(x)

        # Decoding the encoded sequence
        for layer in self.decoder:
            x, _ = layer(x)
        return x

