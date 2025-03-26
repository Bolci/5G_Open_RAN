import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=48):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim if embed_dim % 2 == 0 else embed_dim + 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # Compute the div term
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        pe = pe[:, :embed_dim]  # Remove the extra dimension if embed_dim is odd
        # Add a batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # Register as a buffer (non-trainable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 72, embed_dim: int = 12, num_heads: int = 2, num_layers: int = 1, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        # self.embedding = nn.Linear(input_dim, embed_dim)  # Project input to embed_dim
        self.embedding = TokenEmbedding(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)  # Project back to original dim
        self.dropout = nn.Dropout(dropout)
        self.float()
        self.model_name = "Transformer_AE"

    def forward(self, x):
        x = self.embedding(x)  # Project to embed_dim
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.dropout(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return self.fc_out(decoded)

    def encode(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return self.encoder(x)