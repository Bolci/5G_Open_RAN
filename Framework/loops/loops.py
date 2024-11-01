# This script contains important loops for the development pipeline

import torch
import numpy as np
from copy import copy
import torch.nn as nn


def train_loop(dataloader, model, loss_fn, optimizer, device="cuda"):
    """
    Trains the model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the training data.
        model (torch.nn.Module): The model to be trained.
        loss_fn (Callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (str): The device to use for training (default is "cuda").

    Returns:
        float: The average training loss for the epoch.
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

        # if batch % 1000 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        #     print()

    train_loss /= num_batches
    return train_loss


def valid_loop(dataloader, model, loss_fn, device="cuda", is_train=False):
    """
    Validates the model.

    Args:
        dataloader (torch.utils.data.DataLoader): The data loader for the validation data.
        model (torch.nn.Module): The model to be validated.
        loss_fn (Callable): The loss function.
        device (str): The device to use for validation (default is "cuda").
        is_train (bool): Flag to indicate if the validation is during training (default is False).

    Returns:
        tuple: A tuple containing the average validation loss, a list of losses to print, and a list of loss scores.
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
    # print(test_losses_to_print)
    return test_loss_mean, test_losses_to_print, test_losses_score


def test_loop(dataloader_test, model: nn.Module, loss_fn, threshold: int, device="cuda"):
    """
    Tests the model and calculates classification scores.

    Args:
        dataloader_test (torch.utils.data.DataLoader): The data loader for the test data.
        model (torch.nn.Module): The model to be tested.
        loss_fn (Callable): The loss function.
        threshold (int): The threshold for classification.
        device (str): The device to use for testing (default is "cuda").

    Returns:
        tuple: A tuple containing the classification score for class 0, the classification score for class 1, and a list of predicted results.
    """
    predicted_results = []

    no_samples = len(dataloader_test)
    counter_var_0 = 0
    counter_var_1 = 0

    with torch.no_grad():
        # Iterate over the test dataloader
        for X, y in dataloader_test:
            # Ensure the input tensor has 3 dimensions
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)

            # Perform the forward pass and compute the test loss
            pred = model(X.to(device))
            test_loss = loss_fn(pred, X).item()

            # Update the counter for correctly classified samples for class 0
            if (test_loss <= threshold and y.item() == 0) or (test_loss > threshold and y.item() == 1):
                counter_var_0 += 1

            # Update the counter for correctly classified samples for class 1
            if (test_loss <= threshold and y.item() == 1) or (test_loss > threshold and y.item() == 0):
                counter_var_1 += 1

            # Append the true label and test loss to the predicted results
            predicted_results.append(([copy(y.item()), copy(test_loss)]))

    # Calculate the classification score for class 0
    classification_score_0 = float(counter_var_0) / float(no_samples)
    # Calculate the classification score for class 1
    classification_score_1 = float(counter_var_1) / float(no_samples)

    # Return the classification scores and the predicted results
    return classification_score_0, classification_score_1, predicted_results
