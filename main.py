# Place to work on the optimization of the model

import argparse
from Framework.utils.utils import load_json_as_dict
from Framework.data_workers.data_preprocessor import DataPreprocessor
from Framework.data_workers.data_path_worker import get_all_paths
from Framework.data_workers.data_utils import get_data_loaders, get_datasets
import os

import numpy as np


def main(path, args):

    # Dataset_paths
    all_paths = get_all_paths(path)

    #Data preparing according to preprocessing
    data_preprocessor = DataPreprocessor()
    data_preprocessor.set_cache_path(path["Data_cache_path"])
    data_preprocessor.set_original_seg(path["True_sequence_path"])
    paths_for_datasets = data_preprocessor.preprocess_data(all_paths, 'abs_only')

    #prepare datasets and data_loaders
    datasets = get_datasets(paths_for_datasets)
    dataloaders = get_data_loaders(datasets, args.batch_size)


    '''
    # Load model here
    model = AEFC()

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = RMSELoss()

    # Move model and loss function to device
    model.to(args.device)
    criterion.to(args.device)

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        if epoch % args.log_interval == 0:
            print(
                f"Epoch {epoch}: Train loss {train_loss}, Val loss {val_loss}"
            )
    
    '''


if __name__ == "__main__":
    #wandb.init(project="Anomaly_detection", config={"epochs": 10, "batch_size": 32})
    paths_config = load_json_as_dict('./data_paths.json')

    parser = argparse.ArgumentParser(description="OpenRAN neural network")
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log interval"
    )
    args = parser.parse_args()

    main(paths_config, args)
