import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(62, 56)
        self.fc2 = nn.Linear(56, 48)
        self.fc3 = nn.Linear(48, 40)
        self.fc4 = nn.Linear(40, 32)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# Deep SVDD Model
class DeepSVDD:
    def __init__(self, model, nu=0.1):
        self.model = model
        self.nu = nu  # Fraction of outliers
        self.c = None  # Center of the hypersphere
        self.model_name = "DeepSVDD"
        self.parameters = self.model.parameters

    def to(self, device):
        self.model.to(device)
        if self.c is not None:
            self.c = self.c.to(device)

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

    def forward(self, x):
        return self.model(x)

    # # Deep SVDD Loss function
    def loss_function(self, outputs):
        # Compute distance from center
        dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        # SVDD loss as a combination of inliers and outliers
        return torch.mean(dist) + self.nu * torch.mean(torch.max(dist - torch.mean(dist), torch.zeros_like(dist)))