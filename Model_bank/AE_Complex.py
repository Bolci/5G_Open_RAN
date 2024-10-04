import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


class AEComplex(nn.Module):
    def __init__(self):
        self.model_name = "AE_Complex"

        self.encoder = nn.Sequential(
            nn.Conv1d(self.channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (32, 36)
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (64, 18)
            nn.ReLU(True),
        )

        # Decoder: Upsample back to the original sequence
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),  # (64, 18) -> (32, 36)
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),  # (32, 36) -> (16, 72)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, self.channels, kernel_size=3, stride=1, padding=1),  # (16, 72) -> (2, 144)
            nn.Sigmoid()
        )


def forward(self, x):
        pass