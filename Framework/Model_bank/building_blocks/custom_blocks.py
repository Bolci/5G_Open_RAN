import torch
import torch.nn as nn


class CNN_Block_1D(nn.Module):
    def __init__(self, input_f: int, output_f: int):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(input_f, output_f, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(True),
                                   nn.BatchNorm1d(output_f),
                                   nn.MaxPool1d(2),
                                   )

    def forward(self, x):
        return self.block(x)


class CNN_Block_inv_1D(nn.Module):
    def __init__(self, input_f: int, output_f: int):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose1d(input_f, output_f, kernel_size=3, stride=1, padding=1),
                                   nn.ELU(True),
                                   nn.BatchNorm1d(output_f),
                                   nn.Upsample(scale_factor=2),
                                   )

    def forward(self, x):
        return self.block(x)



