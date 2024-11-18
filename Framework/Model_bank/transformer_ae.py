import torch.nn as nn
import torch

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 72, embed_dim: int = 12, num_heads: int = 2, num_layers: int = 1, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # Project input to embed_dim
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
        x = self.dropout(x)
        encoded = self.encoder(x)
        encoded = self.dropout(encoded)
        decoded = self.decoder(encoded, encoded)
        decoded = self.dropout(decoded)
        return self.fc_out(decoded)