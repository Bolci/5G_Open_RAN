from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

from typing import Optional, Callable
import os
import numpy as np

class DatasetTemplate(Dataset):
    def __init__(self, 
                 data_path: str, 
                 label: Optional = 0, 
                 loader_f: Callable = lambda x: np.load(x), 
                 transform: transforms = None,
                 label_to_one_hot = False):
        
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

        label = np.array([self.label]*loaded_data.shape[1])
        if self.transform:
            data = self.transform(loaded_data)

    
        #label = torch.Tensor(label)
        return loaded_data, loaded_data