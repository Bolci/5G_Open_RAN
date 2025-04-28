import torch.nn as nn


class AEFC(nn.Module):
    """
    Autoencoder model with fully connected layers.
    """

    def __init__(
        self,
    ):
        super(AEFC, self).__init__()
        self.model_name = 'AEFC'
        self.encoder = nn.Sequential(
            nn.Linear(72, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 24),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(24, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 48),
            nn.LeakyReLU(),
            nn.Linear(48, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 72),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
