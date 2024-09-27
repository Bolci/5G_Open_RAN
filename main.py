# Place to work on the optimization of the model

from Framework.dataset import OpenRANDatasetV2
from Framework.loops import train, validate, test
from Framework.metrics import RMSELoss
from Model_bank.models import AEFC
import argparse
import torch
import os
import numpy as np


def run(args):
    # Load dataset here
    true_bts = None
    fake_bts = None
    folders = os.listdir("Data_selection")
    for i, folder in enumerate(folders):
        if "PCI_12" in folder:
            true_bts = (
                torch.cat(
                    (
                        true_bts,
                        torch.load(
                            f"Data_selection/{folder}/channel_responses.pt"
                        ),
                    ),
                    dim=1,
                )
                if i == 0
                else torch.load(
                    f"Data_selection/{folder}/channel_responses.pt"
                )
            )
        elif "PCI_466" in folder:
            fake_bts = (
                torch.cat(
                    (
                        fake_bts,
                        torch.load(
                            f"Data_selection/{folder}/channel_responses.pt"
                        ),
                    ),
                    dim=1,
                )
                if i == 0
                else torch.load(
                    f"Data_selection/{folder}/channel_responses.pt"
                )
            )
    labels = torch.zeros(true_bts.shape[-1])
    labels = torch.cat((labels, torch.ones(fake_bts.shape[1])), dim=0)
    test_data = torch.cat((true_bts, fake_bts), dim=1)

    train_data = torch.load("Data_selection/comeretial/channel_responses.pt")
    # test_data = test_data.unsqueeze(0)
    # train_data = train_data.unsqueeze(0)
    train_set = OpenRANDatasetV2(
        train_data, torch.zeros(train_data.shape[-1]), convert_to_dB=True
    )

    test_sets = OpenRANDatasetV2(test_data, labels, convert_to_dB=True)

    # Data to device
    train_set.to(args.device)
    test_sets.to(args.device)

    # Split dataset for training
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_set, [train_size, val_size]
    )

    # Load data into dataloader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_sets, batch_size=1, shuffle=False
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

    threshold_base = val_loss
    for k in np.arange(1, 3, 0.1):
        threshold = threshold_base * k
        labels, losses, acc = test(model, test_loader, criterion, threshold)
        print(f"Threshold: {threshold}({k} x) has accuracy of {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenRAN neural network")
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--log_interval", type=int, default=1, help="Log interval"
    )
    args = parser.parse_args()
    run(args)
