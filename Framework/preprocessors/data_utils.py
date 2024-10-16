import torch
from torch.utils.data import DataLoader
from .dataset import DatasetTemplate
from copy import copy

def get_datasets(paths: dict, loader_function =  lambda x: torch.load(x)):
    return_datsets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, paths in paths.items():
        for single_path in paths:
            new_dataset = DatasetTemplate(single_path, loader_f=loader_function, load_all_data=True)
            return_datsets[dataset_type].append(copy(new_dataset))

    return return_datsets


def get_data_loaders(Datasets: dict, batch_size):
    return_datasets = {'Train': [], 'Valid': [], 'Test': []}

    for dataset_type, dataset in Datasets.items():
        for single_dataset in dataset:
            bs = batch_size if dataset_type == 'Train' else 1
            return_datasets[dataset_type].append(DataLoader(single_dataset, batch_size=bs, shuffle=True))

    return return_datasets

