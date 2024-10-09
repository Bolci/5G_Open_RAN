import torch.nn as nn
import torch


class RMSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mse = nn.MSELoss(*args, **kwargs)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


