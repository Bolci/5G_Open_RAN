# Place to work on the optimization of the model

from Framework.utils.utils import load_json_as_dict
from Framework.data_workers.data_preprocessor import DataPreprocessor
from Framework.data_workers.data_path_worker import get_all_paths
import os

import numpy as np


def main(path, *args):

    # Dataset_paths
    all_paths = get_all_paths(path)

    data_preprocessor = DataPreprocessor()
    data_preprocessor.set_cache_path(path["Data_cache_path"])
    data_preprocessor.set_original_seg(path["True_sequence_path"])
    data_preprocessor.preprocess_data(all_paths)


    '''

    # Data to device
    train_set.to(args.device)
    # test_sets = test_sets.to(args.device)

    # Split dataset for training
    train_size = int(0.8 * len(train_set))
    test_size = len(train_set) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_set, [train_size, test_size]
    )

    # Load data into dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Additional test sets for thresholding
    # test_loader = torch.utils.data.DataLoader(test_sets, batch_size=args.batch_size, shuffle=False)

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
    print(paths_config)
    '''
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
    '''
    main(paths_config)
