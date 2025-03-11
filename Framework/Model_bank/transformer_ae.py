import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 72, embed_dim: int = 12, num_heads: int = 2, num_layers: int = 1, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # Project input to embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)  # Project back to original dim
        # self.dropout = nn.Dropout(dropout)
        self.float()
        self.model_name = "Transformer_AE"

    def forward(self, x):
        x = self.embedding(x)  # Project to embed_dim
        x = self.positional_encoding(x)  # Add positional encoding
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return self.fc_out(decoded)

    def encode(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return self.encoder(x)