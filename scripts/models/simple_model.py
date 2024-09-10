import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.lay_one = nn.Sequential(*[nn.Identity(72)])

    def forward(self, x):
        return self.lay_one(x)
