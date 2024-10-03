from torch.utils.data import Dataset, DataLoader

import torch

from typing import Optional, Callable
import os
import numpy as np


class DatasetTemplate(Dataset):
    def __init__(self,
                 data_path: str,
                 label: Optional = 0,
                 loader_f: Callable = lambda x: np.load(x),
                 label_extraction_f: Callable = lambda x: int(x.split('_')[2].split('=')[:-1])
                 ):

        self.data_path = data_path
        self.data_names = os.listdir(data_path)
        self.label = label

        self.loader_function = loader_f
        self.label_extraction_function = label_extraction_f

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        label = self.label_extraction_function(data_name)

        data_path = os.path.join(self.data_path, data_name)
        loaded_data = self.loader_function(data_path).astype(np.float32)
        loaded_data = loaded_data.T


        return loaded_data, label
