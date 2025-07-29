# This script contains important loops for the development pipeline

import torch
import numpy as np
from copy import copy
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable

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
        X_ = X.to(device).squeeze()
        pred = model(X_)
        # loss = loss_fn(pred, X_)
        loss = model.loss_function(pred)
        loss = loss.mean()
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
    val_losses_to_print = []
    val_losses_score = []

    with torch.no_grad():
        for X, y in dataloader:
            X_ = X.to(device).squeeze()
            pred = model(X_)
            val_loss = model.loss_function(pred)
            # Uncomment the line below if you want to use a different loss function
            val_losses_score.append(copy(val_loss))

            if not is_train:
                val_losses_to_print.append(torch.vstack([y, copy(val_loss)]))

    val_losses_to_print = torch.concatenate(val_losses_to_print,
                                            dim=1).cpu().numpy().T if val_losses_to_print else []
    val_losses_score = torch.concatenate(val_losses_score, dim=0)
    val_loss_mean = torch.mean(val_losses_score).item()
    val_losses_score = val_losses_score.cpu().numpy()
    print(f"Avg loss: {val_loss_mean:>8f} \n")

    # val_loss_mean - FLOAT - average loss over all samples in the validation set
    # val_losses_to_print - np.ndarray - array of shape [BATCH, 2] with first column being labels and second column being losses
    # val_losses_score - np.ndarray - array of shape [BATCH, ] with per-sample losses
    return val_loss_mean, val_losses_to_print, val_losses_score

def test_loop(dataloader_test, model: nn.Module, loss_fn, predict_class, device="cuda"):
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
    true_labels = []
    predicted_labels = []
    no_samples = 0
    correct_classification_counter = 0

    with torch.no_grad():
        for X, y in dataloader_test:
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)

            pred = model(X.to(device))
            test_loss = model.loss_function(pred)
            # test_loss = loss_fn(pred, X).item()

            predicted_label = model.predict((X.to(device)))
            predicted_labels.append(predicted_label)
            true_labels.append(y)

            predicted_results.append(torch.vstack([copy(y), copy(test_loss)]))

            correct_classification_counter += (y == predicted_label).sum().item()
            no_samples += len(y)

    true_labels = torch.concatenate(true_labels, dim=0).cpu().numpy()
    predicted_labels = torch.concatenate(predicted_labels, dim=0).cpu().numpy()
    accuracy = correct_classification_counter/no_samples
    precision = precision_score(true_labels, predicted_labels, zero_division=0.0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0.0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0.0)

    classification_score = f1
    predicted_results = torch.concatenate(predicted_results, dim=1).cpu().numpy()
    return classification_score, predicted_results, (accuracy, precision, recall, f1)


