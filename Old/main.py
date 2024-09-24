import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from typing import Optional, Callable
import os
import matplotlib.pyplot as plt

from scripts.loops import train_loop, valid_loop, test_loop
from scripts.dataset_template import DatasetTemplate
from scripts.models.autoencoder_cnn import CNNAutoencoder
from scripts.models.autoencoder_lin import LinearAutoencoder
from scripts.models.autoencoder_lstm import LSTMAutoencoder


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


