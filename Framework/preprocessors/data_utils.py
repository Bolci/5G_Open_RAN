import torch
from torch.utils.data import DataLoader
from .dataset import DatasetTemplate
from copy import copy

def get_datasets(paths: dict, loader_function = lambda x: torch.load(x)):
    """
    Creates datasets for training, validation, and testing.

    This function iterates over the provided paths, creates a DatasetTemplate for each path,
    and appends it to the corresponding dataset type (Train, Valid, Test).

    Args:
        paths (dict): A dictionary where keys are dataset types ('Train', 'Valid', 'Test') and values are lists of paths.
        loader_function (Callable): A function to load the data from a file. Defaults to torch.load.

    Returns:
        dict: A dictionary containing lists of DatasetTemplate objects for each dataset type.
    """
    return_datsets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, paths in paths.items():
        for single_path in paths:
            new_dataset = DatasetTemplate(single_path, loader_f=loader_function, load_all_data=True)
            return_datsets[dataset_type].append(copy(new_dataset))

    return return_datsets


def get_data_loaders(Datasets: dict, batch_size):
    """
    Creates data loaders for training, validation, and testing datasets.

    This function iterates over the provided datasets, creates a DataLoader for each dataset,
    and appends it to the corresponding dataset type (Train, Valid, Test).

    Args:
        Datasets (dict): A dictionary where keys are dataset types ('Train', 'Valid', 'Test') and values are lists of DatasetTemplate objects.
        batch_size (int): The batch size to use for the training data loader. Validation and test data loaders use a batch size of 1.

    Returns:
        dict: A dictionary containing lists of DataLoader objects for each dataset type.
    """
    return_datasets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, dataset in Datasets.items():
        for single_dataset in dataset:
            bs = batch_size if dataset_type == 'Train' else 1
            return_datasets[dataset_type].append(DataLoader(single_dataset, batch_size=bs, shuffle=True))

    return return_datasets

