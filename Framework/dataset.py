# Custom dataset class
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

from typing import Optional, Callable
import os
import numpy as np


class OpenRANDatasetV2(Dataset):
    def __init__(self, data_path: str, convert_to_dB: bool = False):
        self.data_path = data_path
        self.data = self.load_data()
        if convert_to_dB:
            self.data = self.to_dB(self.data)

    def load_data(self):
        try:
            data = torch.load(self.data_path, weights_only=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self.data_path} not found")
        return data

    @staticmethod
    def to_dB(data: torch.Tensor) -> torch.Tensor:
        return -20 * torch.log10(torch.abs(data))

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        return self.data[:, idx]

    def to(self, device):
        self.data = self.data.to(device)


class OpenRANDatasetV1(Dataset):
    def __init__(
        self,
        data_path: str,
        label: Optional = 0,
        loader_f: Callable = lambda x: np.load(x),
        transform: transforms = None,
        label_to_one_hot=False,
    ):
        self.data_path = data_path
        self.data_names = os.listdir(data_path)
        self.label = label

        self.transform = transform
        self.loader_function = loader_f

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_path, self.data_names[idx])
        loaded_data = self.loader_function(data_path).astype(np.float32)

        loaded_data = loaded_data.T

        label = np.array([self.label] * loaded_data.shape[1])
        if self.transform:
            data = self.transform(loaded_data)

        v_min, v_max = loaded_data.min(), loaded_data.max()
        new_min, new_max = 0.0, 1.0
        loaded_data = (loaded_data - v_min) / (v_max - v_min) * (
            new_max - new_min
        ) + new_min

        # label = torch.Tensor(label)
        return loaded_data, loaded_data
