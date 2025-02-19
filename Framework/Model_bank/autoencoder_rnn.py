import torch.nn as nn

class RNNAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, unit_type="lstm", dropout=0.2):
        super(RNNAutoencoder, self).__init__()
        self.name = "rnn_ae"
        if unit_type == "lstm":
            self.unit = nn.LSTM
        elif unit_type == "gru":
            self.unit = nn.GRU
        else:
            raise ValueError("Invalid unit type. Choose between 'lstm' and 'gru'")

        self.dropout = dropout
        # Encoder
        self.encoder = nn.Sequential()
        in_dim = input_dim
        for i, size in enumerate(hidden_dims):
            self.encoder.add_module(f"lstm_{i}", self.unit(in_dim, size, batch_first=True))
            in_dim = size

        # Decoder
        self.decoder = nn.Sequential()
        for i, size in enumerate(reversed(hidden_dims)):
            self.decoder.add_module(f"lstm_{i}", self.unit(in_dim, size, batch_first=True))
            in_dim = size
        self.decoder.add_module("output", self.unit(in_dim, input_dim, batch_first=True))

    def forward(self, input):
        # input is of shape [batch, 50, 72]  -- only one feature
        x = input
        # Encoding the input sequence
        for layer in self.encoder:
            x, _ = layer(x)
            x = nn.functional.dropout(x, self.dropout, self.training)

        # Decoding the encoded sequence
        for i, layer in enumerate(self.decoder):
            x, _ = layer(x)
            if i < len(self.decoder) - 1:
                x = nn.functional.dropout(x, self.dropout, self.training)
        return x


