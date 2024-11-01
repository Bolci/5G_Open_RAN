import torch.nn as nn
import torch

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # Project input to embed_dim
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)  # Project back to original dim
        self.float()
        self.model_name = "Transformer_AE"

    def forward(self, x):
        x = self.embedding(x)  # Project to embed_dim
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return self.fc_out(decoded)