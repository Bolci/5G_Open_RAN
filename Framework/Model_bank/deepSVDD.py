import argparse
from Framework.utils.utils import load_json_as_dict, save_txt
from Framework.preprocessors.data_preprocessor import DataPreprocessor
from Framework.preprocessors.data_path_worker import get_all_paths
from Framework.preprocessors.data_utils import get_data_loaders, get_datasets
from Framework.metrics.metrics import RMSELoss
from Framework.Model_bank.autoencoder_cnn import CNNAutoencoder, CNNAutoencoderV2
from Framework.loops.loops import train_loop, valid_loop, test_loop
from Framework.postprocessors.postprocessor_functions import plot_data_by_labels, mean_labels_over_epochs
from Framework.postprocessors.postprocesor import Postprocessor
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(72, 64)
        self.fc2 = nn.Linear(64, 56)
        self.fc3 = nn.Linear(56, 48)
        self.fc4 = nn.Linear(48, 40)

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

    # Deep SVDD Loss function
    def loss_function(self, outputs):
        # Compute distance from center
        dist = torch.sum((outputs - self.c) ** 2, dim=-1)
        # SVDD loss as a combination of inliers and outliers
        return torch.mean(dist) + self.nu * torch.mean(torch.max(dist - torch.mean(dist), torch.zeros_like(dist)))

