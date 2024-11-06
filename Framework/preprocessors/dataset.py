from calendar import day_abbr

from torch.utils.data import Dataset
import torch
from typing import Optional, Callable
import os
import numpy as np
import time
from copy import copy

class DatasetTemplate(Dataset):
    """
    A custom dataset template for loading and processing data.

    Attributes:
        device (str): The device to load the data onto (e.g., 'cuda' or 'cpu').
        data_path (str): The path to the directory containing the data files.
        data_names (list): List of data file names in the data_path directory.
        label (Optional): The label for the dataset.
        loader_function (Callable): Function to load the data from a file.
        label_extraction_function (Callable): Function to extract the label from a file name.
        load_all_data (bool): Flag to indicate whether to load all data into memory.
        data (torch.Tensor): Tensor containing all loaded data (if load_all_data is True).
        labels (torch.Tensor): Tensor containing all labels (if load_all_data is True).
    """

    def __init__(self,
                 data_path: str,
                 label: Optional = 0,
                 loader_f: Callable = lambda x: np.load(x),
                 label_extraction_f: Callable = lambda x: int((x.split('_')[2]).split('=')[1][:-3]),
                 load_all_data: bool = False,
                 device: str = "cuda",
                 ):
        """
        Initializes the DatasetTemplate.

        Args:
            data_path (str): The path to the directory containing the data files.
            label (Optional): The label for the dataset.
            loader_f (Callable): Function to load the data from a file.
            label_extraction_f (Callable): Function to extract the label from a file name.
            load_all_data (bool): Flag to indicate whether to load all data into memory.
            device (str): The device to load the data onto (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self.data_path = data_path
        self.data_names = os.listdir(data_path)
        self.label = label
        self.loader_function = loader_f
        self.label_extraction_function = label_extraction_f

        self.load_all_data = load_all_data
        if self.load_all_data:
            self.data = None
            self.labels = None
            self.load_data()

    def load_data(self):
        """
        Loads all data and labels into memory and normalizes the data.
        """
        for data_name in self.data_names:
            data_path = os.path.join(self.data_path, data_name)
            loaded_data = self.loader_function(data_path).float()
            loaded_data = loaded_data.permute(0, 2, 1)

            v_min, v_max = loaded_data.min(dim=-1, keepdim=True).values, loaded_data.max(dim=-1, keepdim=True).values
            new_min, new_max = 0.0, 1.0
            loaded_data = (loaded_data - v_min) / (v_max - v_min) * (new_max - new_min) + new_min
            label = self.label_extraction_function(data_name)

            if self.data is None:
                self.data = loaded_data
            else:
                self.data = torch.cat((self.data, loaded_data), dim=0)

            if self.labels is None:
                self.labels = torch.tensor([label]*len(loaded_data))
            else:
                self.labels = torch.cat((self.labels, torch.tensor([copy(label)]*len(loaded_data))), dim=0)
        self.data = self.data.to(self.device)
        self.labels = self.labels.to(self.device)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        if self.load_all_data:
            return len(self.data)
        else:
            return len(self.data_names)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its label by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the data and the label.
        """
        if self.load_all_data:
            data = self.data[idx, :, :]
            label = self.labels[idx]
            return data, label
        else:
            data_name = self.data_names[idx]
            label = self.label_extraction_function(data_name)

            data_path = os.path.join(self.data_path, data_name)
            loaded_data = self.loader_function(data_path).float()
            loaded_data = loaded_data.T

            v_min, v_max = loaded_data.min(), loaded_data.max()
            new_min, new_max = 0.0, 1.0
            loaded_data = (loaded_data - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

            return loaded_data, label
