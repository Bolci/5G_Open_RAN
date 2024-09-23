import torch
import argparse
from models import *
def train_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module) -> torch.nn.Module:
    """
    Train the model on the dataset for the specified number of epochs.
    Args:
    model (torch.nn.Module): Model to train.
    dataloader (torch.data.DataLoader): DataLoader with the dataset.
    optimizer (torch.optim.optimizer): Optimizer to use.
    loss_fn (torch.nn.Module): Loss function to use.


    Returns:
    torch.nn.Module: Trained model.
    """

    model.train()
    total_loss = 0
    for x in dataloader:
        optimizer.zero_grad()
        output = model(x)
        loss = torch.sqrt(loss_fn(output, x))
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def validate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> float:
    """
    Validate the model on the dataset.
    Args:
    model (torch.nn.Module): Model to validate.
    dataloader (torch.data.DataLoader): DataLoader with the dataset.
    loss_fn (torch.nn.Module): Loss function to use.


    Returns:
    float: Loss on the dataset.
    """

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            output = model(x)
            loss = torch.sqrt(loss_fn(output, x))
            total_loss += loss.item()
    return total_loss / len(dataloader)



def run(args: argparse.Namespace):
    """
    Run the training process.
    Args:
    args (argparse.Namespace): Arguments to use.
    """
    lr = args.learning_rate
    epochs = args.epochs
    batch = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = AE1DCNN()
    model = AEFC()
    model.to(device)

    dataset = torch.load("Data_selection/commercial_abs_v2.pt")
    dataset = dataset.reshape((dataset.shape[0], -1)).T
    # dataset = dataset.unsqueeze(1)
    dataset = dataset.to(device)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()


    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        test_loss = validate_model(model, test_loader, loss_fn)
        print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")

    torch.save(model, "model.pt")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", default=1000, type=int)
    args.add_argument("--learning_rate", default=0.001, type=float)
    args.add_argument("--batch_size", default=10000, type=int)
    args = args.parse_args()
    run(args)


