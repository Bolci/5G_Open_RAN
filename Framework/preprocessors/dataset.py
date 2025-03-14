from calendar import day_abbr

from torch.utils.data import Dataset
import torch
from typing import Optional, Callable
import os
import numpy as np
import time
from copy import copy

class DatasetTemplate(Dataset):
    def __init__(self,
             data_path: str,
             label: Optional = 0,
             loader_f: Callable = lambda x: np.load(x),
             label_extraction_f: Callable = lambda x: int((x.split('_')[-1]).split('=')[1][:-3]),
             load_all_data: bool = False,
             device: str = "cuda",
             ):
        """
        Initializes the DatasetTemplate object.

        Args:
            data_path (str): The path to the directory containing the data files.
            label (Optional): The default label to use if no label extraction function is provided. Defaults to 0.
            loader_f (Callable): A function to load the data from a file. Defaults to loading a numpy array.
            label_extraction_f (Callable): A function to extract the label from the data file name. Defaults to extracting an integer from the file name.
            load_all_data (bool): Whether to load all data into memory at once. Defaults to False.
            device (str): The device to load the data onto. Defaults to "cuda".
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
        Loads data from the specified data path and processes it.

        This method iterates over all data files in the data path, loads each file using the provided loader function,
        normalizes the data, and extracts labels using the provided label extraction function. The loaded data and labels
        are then concatenated to form a single dataset.

        Attributes:
            self.data (torch.Tensor): The concatenated data from all files.
            self.labels (torch.Tensor): The concatenated labels from all files.
        """
        for data_name in self.data_names:
            # Construct the full path to the data file
            data_path = os.path.join(self.data_path, data_name)

            # Load the data using the loader function and convert to float
            loaded_data = self.loader_function(data_path).float()

            # Permute the dimensions of the loaded data
            loaded_data = loaded_data.permute(0, 2, 1)

            # Normalize the data to the range [0, 1]
            v_min, v_max = loaded_data.min(dim=-1, keepdim=True).values, loaded_data.max(dim=-1, keepdim=True).values
            new_min, new_max = 0.0, 1.0
            loaded_data = (loaded_data - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

            # Extract the label from the data file name
            label = self.label_extraction_function(data_name)

            # Concatenate the loaded data to the existing data
            if self.data is None:
                self.data = loaded_data
            else:
                self.data = torch.cat((self.data, loaded_data), dim=0)

            # Concatenate the labels to the existing labels
            if self.labels is None:
                self.labels = torch.tensor([label]*len(loaded_data))
            else:
                self.labels = torch.cat((self.labels, torch.tensor([copy(label)]*len(loaded_data))), dim=0)

        # Move the data and labels to the specified device
        self.data = self.data.to(self.device)
        self.labels = self.labels.to(self.device)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        If all data is loaded into memory, it returns the length of the data tensor.
        Otherwise, it returns the number of data files.

        Returns:
            int: The number of samples in the dataset.
        """
        if self.load_all_data:
            return len(self.data)
        else:
            return len(self.data_names)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        If all data is loaded into memory, it returns the data and label at the specified index.
        Otherwise, it loads the data from the file, normalizes it, and returns the data and label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the data and the label.
        """
        if self.load_all_data:
            # Retrieve data and label from memory
            data = self.data[idx, :, :]
            label = self.labels[idx]
            return data, label

        else:
            # Retrieve data file name and extract label
            data_name = self.data_names[idx]
            label = self.label_extraction_function(data_name)

            # Construct the full path to the data file
            data_path = os.path.join(self.data_path, data_name)

            # Load the data using the loader function and convert to float
            loaded_data = self.loader_function(data_path).float()

            # Transpose the loaded data
            loaded_data = loaded_data.T

            # Normalize the data to the range [0, 1]
            v_min, v_max = loaded_data.min(), loaded_data.max()
            new_min, new_max = 0.0, 1.0
            loaded_data = (loaded_data - v_min) / (v_max - v_min) * (new_max - new_min) + new_min

            return loaded_data, label