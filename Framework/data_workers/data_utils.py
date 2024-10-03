from torch.utils.data import Dataset, DataLoader
from .dataset import DatasetTemplate
from copy import copy

def get_datasets(paths: dict):
    return_datsets = {'Train': [], 'Valid': [], 'Test': []}
    for dataset_type, paths in paths.items():

        for single_path in paths:
            new_dataset = DatasetTemplate(single_path)
            return_datsets[dataset_type].append(copy(new_dataset))

    return return_datsets


def get_data_loaders(Datasets):
    pass
