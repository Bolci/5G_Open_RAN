import torch
import torch.nn as nn
from aiohttp.web_routedef import static


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=2),
            # nn.ReLU(True),
            # nn.MaxPool1d(2),
            # nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=2),
            # nn.ReLU(True),
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (32, 36)
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (64, 18)
            nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (128, 9)
            nn.ReLU(True),
        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),  # (128, 9) -> (64, 18)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (32, 36)
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (16, 72)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (2, 144)
            nn.ReLU()
        )
        # self.decoder = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),  # (128, 9) -> (64, 18)
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (32, 36)
        #     nn.ReLU(True),
        #     nn.Upsample(scale_factor=2),
        #     nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (16, 72)
        #     nn.ReLU(True),
        #     nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (2, 144)
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        # for layer in self.layers:
        #     x = layer(x)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return (decoded - x)**2

# Deep SVDD Model
class DeepSVDD(nn.Module):
    def __init__(self, model, nu=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.nu = nu  # Fraction of outliers
        self.c = None  # Center of the hypersphere
        self.radius = 1
        self.name = "DeepSVDD"

    def to(self, device):
        self.model.to(device)
        return self

    # Initialize center c as the mean of the initial data points
    def initialize_center(self, data_loader, device):
        n_samples = 0
        with torch.no_grad():
            sample_output = self.model(torch.zeros(1, 1, 72, device=device))
        c = torch.zeros(sample_output.shape[1:], device=device)
        self.model.eval()

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                outputs = self.model(x)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples

        # Avoid center being too close to zero
        c[(abs(c) < 1e-6)] = 1e-6

        self.c = c

    def update_center(self, data_loader, device):
        self.initialize_center(data_loader, device)

    @staticmethod
    def get_radius(dist: torch.Tensor, nu: float):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return torch.quantile(torch.sqrt(dist), 1 - nu)

    def forward(self, x):
        return self.model(x), self.c, self.nu, self.radius



if __name__ == '__main__':
    x = torch.randn(1, 1, 72)
    model = SimpleNet()
    model(x)


