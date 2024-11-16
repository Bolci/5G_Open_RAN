import torch
from torch.utils.data import DataLoader
from .dataset import DatasetTemplate
from copy import copy

def get_datasets(paths: dict, loader_function = lambda x: torch.load(x)):
    """
    Create datasets for training, validation, and testing from the given paths.

    Args:
        paths (dict): A dictionary containing paths for 'Train', 'Valid', and 'Test' datasets.
        loader_function (function): A function to load the data from the given path. Defaults to torch.load.

    Returns:
        dict: A dictionary with keys 'Train', 'Valid', and 'Test', each containing a list of DatasetTemplate objects.
    """
    return_datsets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, paths in paths.items():
        for single_path in paths:
            new_dataset = DatasetTemplate(single_path, loader_f=loader_function, load_all_data=True)
            return_datsets[dataset_type].append(copy(new_dataset))

    return return_datsets


def get_data_loaders(Datasets: dict, batch_size):
    """
    Create data loaders for training, validation, and testing datasets.

    Args:
        Datasets (dict): A dictionary containing 'Train', 'Valid', and 'Test' datasets.
        batch_size (int): The batch size to use for the data loaders.

    Returns:
        dict: A dictionary with keys 'Train', 'Valid', and 'Test', each containing a list of DataLoader objects.
    """
    return_datasets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, dataset in Datasets.items():
        for single_dataset in dataset:
            bs = batch_size if dataset_type == 'Train' else 1
            return_datasets[dataset_type].append(DataLoader(single_dataset, batch_size=bs, shuffle=True))

    return return_datasets
