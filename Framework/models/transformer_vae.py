import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

class TransformerVAE(nn.Module):
    def __init__(self, input_dim: int = 72, embed_dim: int = 12, num_heads: int = 2, num_layers: int = 1, dropout=0.1):
        super(TransformerVAE, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = embed_dim * 2  # Latent space size

        # Encoder
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Latent space
        self.fc_mu = nn.Linear(embed_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(embed_dim, self.latent_dim)

        # Decoder
        self.fc_latent = nn.Linear(self.latent_dim, embed_dim)  # Project back to embedding dim
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)

        self.model_name = "Transformer_vae"
        self.float()

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from N(mu, var)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        batch, seq_len, features = x.size()

        use_fft = False
        add_temp_dev = False
        add_first_derivative = False
        if use_fft:
            x_freq = torch.fft.rfft(x, dim=1)  # Outputs complex numbers (you might take magnitude/phase as features)
            x= torch.abs(x_freq)

        if add_temp_dev:
            temporal_std = x.std(dim=1, keepdim=True)  # (B, 1, F)
            x = torch.cat([x, temporal_std.expand(-1, 48, -1)], dim=2)  # (B, 48, 73)
        if add_first_derivative:
            delta = x[:, 1:, :] - x[:, :-1, :]  # (B, 47, 72)
            delta = F.pad(delta, (0, 0, 1, 0))  # pad first time step → (B, 48, 72)
            x = torch.cat([x, delta], dim=2)  # (B, 48, 144)

        x = self.embedding(x)
        x = self.positional_encoding(x)  # Add positional encoding
        # x = self.dropout(x)
        encoded = self.encoder(x)  # shape: (batch_size, seq_len, embed_dim)
        encoded = encoded.mean(dim=1)
        # encoded = self.pooling(encoded)  # shape: (batch_size, embed_dim)

        # Latent space
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        z = self.reparameterize(mu, log_var)

        # Decode
        z = self.fc_latent(z)

        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoded = self.decoder(z, z)
        decoded = self.fc_out(decoded)

        if use_fft:
            decoded = torch.fft.irfft(decoded, n=seq_len, dim=1)

        if add_temp_dev or add_first_derivative:
            decoded = decoded[:, :, :features]

        return decoded, mu, log_var