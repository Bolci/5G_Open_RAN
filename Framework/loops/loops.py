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
        X_ = X.to(device)
        pred = model(X_)
        loss = loss_fn(pred, X_)
        loss = loss.mean()  # Ensure loss is a scalar
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
            X = X.to(device)
            pred = model(X)
            test_loss = loss_fn(pred, X) # return per sample loss
            test_loss = test_loss.mean(dim=(1,2))  #
            test_losses_score.append(copy(test_loss))

            if not is_train:
                test_losses_to_print.append(torch.vstack([y, copy(test_loss)]))

    test_losses_to_print = torch.concatenate(test_losses_to_print, dim=0).cpu().numpy() if test_losses_to_print else []
    test_losses_score = torch.concatenate(test_losses_score, dim=0)
    test_loss_mean = torch.mean(test_losses_score).item()
    test_losses_score = test_losses_score.cpu().numpy()
    print(f"Avg loss: {test_loss_mean:>8f} \n")
    return test_loss_mean, test_losses_to_print, test_losses_score

def test_loop(dataloader_test,
              model: nn.Module,
              loss_fn: Callable,
              predict_class: Callable,
              device="cuda"):
    predicted_results = []
    true_labels = []
    predicted_labels = []
    correct_classification_counter = 0
    no_samples = len(dataloader_test)

    with torch.no_grad():
        for X, y in dataloader_test:
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)

            pred = model(X.to(device))
            test_loss = loss_fn(pred, X) # return per sample loss
            test_loss = test_loss.mean(dim=(1,2))
            predicted_label = predict_class(test_loss)
            predicted_labels.append(predicted_label)
            true_labels.append(y)

            # if y == predict_class(test_loss):
            #     correct_classification_counter += 1

            predicted_results.append(torch.vstack([copy(y), copy(test_loss)]))
    true_labels = torch.concatenate(true_labels, dim=0).cpu().numpy()
    predicted_labels = torch.concatenate(predicted_labels, dim=0).cpu().numpy()
    # classification_score = correct_classification_counter/no_samples
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    classification_score = f1
    predicted_results = torch.concatenate(predicted_results, dim=0).cpu().numpy()
    return classification_score, predicted_results, (precision, recall, f1)
