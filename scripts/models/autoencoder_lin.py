import torch
import torch.nn as nn


class LinearAutoencoder(nn.Module):
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        
        # Encoder: Flatten (2, 144) to 288, and progressively reduce to a latent space
        self.encoder = nn.Sequential(
            nn.Linear(2 * 144, 128),  # 288 -> 128
            nn.ReLU(True),
            nn.Linear(128, 64),       # 128 -> 64
            nn.ReLU(True),
            nn.Linear(64, 32),        # 64 -> 32 (latent space)
            nn.ReLU(True)
        )
        
        # Decoder: Reconstruct back from latent space to (2, 144)
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),        # 32 -> 64
            nn.ReLU(True),
            nn.Linear(64, 128),       # 64 -> 128
            nn.ReLU(True),
            nn.Linear(128, 2 * 144),  # 128 -> 288 (2, 144)
        )
    
    def forward(self, x):
        # Flatten the input (batch_size, 2, 144) -> (batch_size, 288)
        x = x.view(x.size(0), -1)
        
        # Pass through the encoder and decoder
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Reshape the output back to (batch_size, 2, 144)
        x = x.view(x.size(0), 2, 144)
        
        return x