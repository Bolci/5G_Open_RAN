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
                 label_extraction_f: Callable = lambda x: int((x.split('_')[2]).split('=')[1][:-3])
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
        loaded_data = self.loader_function(data_path).float()
        loaded_data = loaded_data.T

        v_min, v_max = loaded_data.min(), loaded_data.max()
        new_min, new_max = 0.0, 1.0
        loaded_data = (loaded_data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min

        return loaded_data, label
