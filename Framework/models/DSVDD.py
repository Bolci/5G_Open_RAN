import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from Framework.models.transformer_ae import PositionalEncoding
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# Deep SVDD Model
class base_DSVDD(nn.Module):
    def __init__(self, *args, **kwargs):
        super(base_DSVDD, self).__init__()
        self.init_model(*args, **kwargs)
        self.nu = 0.01  # Fraction of outliers
        # Center and radius parameters
        self.register_buffer('c', None)  # Center [out_features]
        self.R = nn.Parameter(torch.ones(self.projection.out_features), requires_grad=True)# Per-dimension radii [out_features]
        self.model_name = "DeepSVDD"

    @abstractmethod
    def init_model(self, *args, **kwargs):
        """
        Initialize the model architecture.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    # Initialize center c as the mean of the initial data points
    def initialize_center(self, data_loader, device):
        n_samples = 0
        c = torch.zeros(self.projection.out_features, device=device)
        self.eval()

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                outputs = self.forward(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
            c /= n_samples

            # Avoid center being too close to zero
            c[(abs(c) < 1e-6)] = 1e-6
            self.c = c

            # Second pass: compute radius R
            all_outputs  = []
            for x, _ in data_loader:
                x = x.to(device)
                outputs = self.forward(x)
                all_outputs.append(outputs)
            all_outputs = torch.cat(all_outputs, dim=0)
            # Compute MAD (median absolute deviation) for each dimension
            abs_dev = torch.abs(all_outputs - c.unsqueeze(0))
            mad = torch.median(abs_dev, dim=0).values
            # Use 1.4826 * MAD as robust estimate of standard deviation
            self.R.data = 1.4826 * mad
            self.R.data = torch.clamp(self.R, min=1e-6)  # Ensure no zero radii

            print(f"Initialized center with shape {self.c.shape}")
            print(f"Radius stats - Mean: {self.R.mean():.4f}, "
                  f"Std: {self.R.std():.4f}, "
                  f"Min: {self.R.min():.4f}, "
                  f"Max: {self.R.max():.4f}")
    # def forward(self, x):
    #     return self.model(x)

    def get_anomaly_scores(self, x):
        """
        Compute anomaly scores using Mahalanobis-like distance
        with per-dimension radii.
        """
        outputs = self.forward(x)  # [batch_size, out_features]

        # Normalized squared distances per dimension
        # (x_i - c_i)^2 / R_i^2
        normalized_sq_dist = (outputs - self.c.unsqueeze(0)) ** 2 / (self.R ** 2 + 1e-6)

        # Sum over dimensions to get anomaly score
        return torch.sum(normalized_sq_dist, dim=1)  # [batch_size]
    def predict(self, x, threshold_ratio=1.0):
        """
        Predict if samples are anomalies.

        Args:
            x: Input tensor
            threshold_ratio: Multiplier for the radius threshold (default 1.0)

        Returns:
            predictions: 1 for anomalies, 0 for normal
            scores: Anomaly scores
        """
        scores = self.get_anomaly_scores(x)
        predictions = (scores > threshold_ratio).int()
        return predictions
    def loss_function(self, outputs):
        """
        Compute the DSVDD loss with per-dimension radii.

        The loss encourages:
        1. Points to be close to the center in each dimension
        2. The radii to be as small as possible (regularization)
        """
        # Normalized squared distances [batch_size, out_features]
        normalized_sq_dist = (outputs - self.c.unsqueeze(0)) ** 2 / (self.R ** 2 + 1e-6)

        # Per-sample loss: sum over dimensions
        per_sample_loss = torch.sum(normalized_sq_dist, dim=1)  # [batch_size]

        # Add regularization to prevent radii from becoming too large
        radius_reg = torch.sum(torch.clamp(self.R, min=1e-6))

        # Final loss
        loss = per_sample_loss + 0.5 * radius_reg / self.projection.out_features

        return loss

class RNN_DSVDD(base_DSVDD):
    def init_model(self, input_dim=62, hidden_dim=12, num_layers=1, dropout=0.1, cell_type='LSTM'):
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)

        self.projection = nn.Linear(hidden_dim, hidden_dim)  # Project to latent space
    def forward(self, x):
        """
        Forward pass of the RNN_DSVDD model.
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
        Returns:
            Output tensor of shape [batch, hidden_dim]
        """
        # Apply RNN
        x, _ = self.rnn(x)
        # Take the last output of the RNN
        x = x[:, -1, :]
        # Project to latent space
        # z = self.projection(x)
        return x

class CNN_DSVDD(base_DSVDD):
    def init_model(self, input_dim=62, out_features=8, num_channels=8, kernel_size=5, dropout=0.1):
        # Reshape input to (batch, channels, features) for 1D convolution
        # Assuming input shape: (batch_size, seq_len, input_dim)
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels=input_dim, out_channels=num_channels,
                      kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second conv block
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels * 2,
                      kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(num_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Third conv block
            nn.Conv1d(in_channels=num_channels * 2, out_channels=num_channels * 4,
                      kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(num_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final projection to latent space
        self.projection = nn.Linear(num_channels * 4, out_features)

        # Global average pooling over time
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        """
        Forward pass of the CNN_DSVDD model.
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
        Returns:
            Output tensor of shape [batch, out_features]
        """
        # x shape: (batch, time, features)
        # Permute to (batch, features, time) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x

class Transformer_DSVDD(base_DSVDD):
    def init_model(self, input_dim=62, out_features=12, num_heads=2, num_layers=1, dropout=0.1):
        self.embedding = nn.Linear(input_dim, out_features)  # Project input to out_features
        self.positional_encoding = PositionalEncoding(out_features)
        encoder_layers = TransformerEncoderLayer(d_model=out_features, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.projection = nn.Linear(out_features, out_features)  # Project to latent space
    def forward(self, x):
        """
        Forward pass of the Transformer_DSVDD model.
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
        Returns:
            Output tensor of shape [batch, out_features]
        """
        x = self.embedding(x)  # Project to out_features
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.transformer_encoder(x)
        # Take the mean over the sequence length dimension
        x = torch.mean(x, dim=1)
        x = self.projection(x)
        return x