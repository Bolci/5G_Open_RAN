import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(72, 64)
        self.fc2 = nn.Linear(64, 56)
        self.fc3 = nn.Linear(56, 48)
        self.fc4 = nn.Linear(48, 40)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.ac(self.fc4(x))
        return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, input_dim: int = 72, embed_dim: int = 12, num_heads: int = 2, num_layers: int = 1, dropout=0.1):
#         super(TransformerEncoder, self).__init__()
#         self.embedding = nn.Linear(input_dim, embed_dim)  # Project input to embed_dim
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#         self.dropout = nn.Dropout(dropout)
#         self.float()
#         self.model_name = "Transformer_E"
#
#     def forward(self, x):
#         x = self.embedding(x)  # Project to embed_dim
#         x = self.dropout(x)
#         encoded = self.encoder(x)
#         return encoded
# Deep SVDD Model
class DeepSVDD(nn.Module):
    def __init__(self, model, nu=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.nu = nu  # Fraction of outliers
        self.c = None  # Center of the hypersphere
        self.radius = 99
        self.name = "DeepSVDD"

    def to(self, device):
        self.model.to(device)
        return self

    # Initialize center c as the mean of the initial data points
    def initialize_center(self, data_loader, device):
        n_samples = 0
        c = torch.zeros(self.model.fc4.out_features, device=device)
        self.model.eval()

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device).squeeze()
                outputs = self.model(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        # Avoid center being too close to zero
        c[(abs(c) < 1e-6)] = 1e-6

        self.c = c

    def update_center(self, data_loader, device):
        self.initialize_center(data_loader, device)

    def forward(self, x):
        return self.model(x), self.c, self.nu
