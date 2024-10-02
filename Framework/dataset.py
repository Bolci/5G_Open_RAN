# Custom dataset class
from torch.utils.data import Dataset

import torch

from typing import Optional, Callable
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch

from typing import Optional, Callable
import os
import numpy as np


class DatasetTemplate(Dataset):
    def __init__(self,
                 data_path: str,
                 label: Optional = 0,
                 loader_f: Callable = lambda x: np.load(x)
                 ):
        self.data_path = data_path
        self.data_names = os.listdir(data_path)
        self.label = label

        self.loader_function = loader_f

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_path, self.data_names[idx])
        loaded_data = self.loader_function(data_path).astype(np.float32)

        loaded_data = loaded_data.T

        label = np.array([self.label] * loaded_data.shape[1])


        # v_min, v_max = loaded_data.min(), loaded_data.max()
        # new_min, new_max = 0.0, 1.0
        # loaded_data = (loaded_data - v_min)/(v_max - v_min)*(new_max - new_min) + new_min

        # label = torch.Tensor(label)
        return loaded_data, loaded_data
