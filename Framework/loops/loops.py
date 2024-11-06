import torch
import numpy as np
from copy import copy
import torch.nn as nn

def train_loop(dataloader, model, loss_fn, optimizer, device="cuda"):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        The DataLoader for the training data.
    model : nn.Module
        The model to be trained.
    loss_fn : Callable
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    device : str, optional
        The device to use for training (default is "cuda").

    Returns
    -------
    float
        The average training loss.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X_ = X.to(device)
        pred = model(X_)
        loss = loss_fn(pred, X_)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print()

    train_loss /= num_batches
    return train_loss

def valid_loop(dataloader, model, loss_fn, device="cuda", is_train=False):
    """
    Validates the model.

    Parameters
    ----------
    dataloader : DataLoader
        The DataLoader for the validation data.
    model : nn.Module
        The model to be validated.
    loss_fn : Callable
        The loss function.
    device : str, optional
        The device to use for validation (default is "cuda").
    is_train : bool, optional
        Whether the validation is being done during training (default is False).

    Returns
    -------
    tuple
        The mean validation loss, a list of losses to print, and a list of score losses.
    """
    test_losses_to_print = []
    test_losses_score = []

    with torch.no_grad():
        for X, y in dataloader:
            X_ = X.to(device)
            pred = model(X_)
            test_loss = loss_fn(pred, X_).item()
            test_losses_score.append(copy(test_loss))

            if not is_train:
                test_losses_to_print.append([copy(y.item()), copy(test_loss)])

    test_loss_mean = np.mean(np.asarray(test_losses_score))
    print(f"Avg loss: {test_loss_mean:>8f} \n")

    return test_loss_mean, test_losses_to_print, test_losses_score

def test_loop(dataloader_test, model: nn.Module, loss_fn, threshold: int, device="cuda"):
    """
    Tests the model.

    Parameters
    ----------
    dataloader_test : DataLoader
        The DataLoader for the test data.
    model : nn.Module
        The model to be tested.
    loss_fn : Callable
        The loss function.
    threshold : int
        The threshold for classification.
    device : str, optional
        The device to use for testing (default is "cuda").

    Returns
    -------
    tuple
        The classification score and a list of predicted results.
    """
    predicted_results = []

    no_samples = len(dataloader_test)
    counter_var_0 = 0
    #counter_var_1 = 0

    with torch.no_grad():
        for X, y in dataloader_test:
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)

            pred = model(X.to(device))
            test_loss = loss_fn(pred, X).item()

            if (test_loss <= threshold and y.item() == 0) or (test_loss > threshold and y.item() == 1):
                counter_var_0 +=1

            #if (test_loss <= threshold and y.item() == 1) or (test_loss > threshold and y.item() == 0):
            #    counter_var_1 +=1

            predicted_results.append(([copy(y.item()), copy(test_loss)]))
    classification_score_0 = float(counter_var_0)/float(no_samples)
    #classification_score_1 = float(counter_var_1) / float(no_samples)

    #return classification_score_0, classification_score_1, predicted_results
    return classification_score_0, predicted_results
