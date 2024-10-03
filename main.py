# Place to work on the optimization of the model

import argparse
from Framework.utils.utils import load_json_as_dict
from Framework.data_workers.data_preprocessor import DataPreprocessor
from Framework.data_workers.data_path_worker import get_all_paths
from Framework.data_workers.data_utils import get_data_loaders, get_datasets
from Framework.metrics.metrics import RMSELoss
from Model_bank.autoencoder_cnn import AEFC
from Framework.loops import train_loop, valid_loop, test_loop
import os
import torch
import torch.nn as nn
import numpy as np


def train_with_hp_setup(datasets, batch_size, learning_rate, no_epochs, device):
    dataloaders = get_data_loaders(datasets, batch_size)
    model = AEFC()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = RMSELoss()
    #criterion_with_reduction = RMSELoss(reduction='none')

    model.to(device)
    criterion.to(device)

    #do one validation loop to ini everything
    valid_loss_mean, valid_loss_per_sample = valid_loop(dataloaders['Valid'][0],
                                                        model,
                                                        criterion,
                                                        device=device)

    train_loss_mean = []
    valid_loss_mean = []
    for epoch in range(no_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loss = train_loop(dataloaders['Train'][0],
                                model,
                                criterion,
                                optimizer,
                                device=device)

        valid_loss_true, valid_metric_true = valid_loop(dataloaders['Valid'][0],
                                                        model,
                                                        criterion,
                                                        device=device)





def main(path, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset_paths
    all_paths = get_all_paths(path)

    #Data preparing according to preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.set_cache_path(path["Data_cache_path"])
    data_preprocessor.set_original_seg(path["True_sequence_path"])
    paths_for_datasets = data_preprocessor.preprocess_data(all_paths, 'abs_only', preprocessing_performed = True)

    #prepare datasets and data_loaders
    datasets = get_datasets(paths_for_datasets)

    train_with_hp_setup(datasets, args.batch_size, args.learning_rate, args.epochs, device)



if __name__ == "__main__":
    #wandb.init(project="Anomaly_detection", config={"epochs": 10, "batch_size": 32})
    paths_config = load_json_as_dict('./data_paths.json')

    parser = argparse.ArgumentParser(description="OpenRAN neural network")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log interval"
    )
    args = parser.parse_args()

    main(paths_config, args)
