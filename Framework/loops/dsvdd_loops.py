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
        pred = model.model(X_)
        # loss = loss_fn(pred, X_)
        loss = model.loss_function(pred)
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
    test_losses_to_print = []
    test_losses_score = []

    with torch.no_grad():
        for X, y in dataloader:
            X_ = X.to(device).squeeze()
            pred = model.model(X_)
            test_loss = model.loss_function(pred).item()

            # test_loss = loss_fn(pred, X_).item()
            test_losses_score.append(copy(test_loss))

            if not is_train:
                test_losses_to_print.append([copy(y.item()), copy(test_loss)])

    test_loss_mean = np.mean(np.asarray(test_losses_score))
    print(f"Avg loss: {test_loss_mean:>8f} \n")
    # print(test_losses_to_print)
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
    true_labels = []
    predicted_labels = []
    no_samples = len(dataloader_test)
    counter_var_0 = 0

    with torch.no_grad():
        for X, y in dataloader_test:
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)

            pred = model.model(X.to(device))
            test_loss = model.loss_function(pred).item()
            # test_loss = loss_fn(pred, X).item()

            predicted_label = 1 if test_loss > threshold else 0
            predicted_labels.append(predicted_label)
            true_labels.append(y.item())

            if (test_loss <= threshold and y.item() == 0) or (test_loss > threshold and y.item() == 1):
                counter_var_0 +=1


            predicted_results.append(([copy(y.item()), copy(test_loss)]))
    classification_score_0 = float(counter_var_0)/float(no_samples)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    classification_score_0 = f1
    return classification_score_0, predicted_results, (precision, recall, f1)


def test_loop_general(dataloader_test,
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
            test_loss = loss_fn(pred, X).item()

            predicted_label = predict_class(test_loss)
            predicted_labels.append(predicted_label)
            true_labels.append(y.item())

            if y == predict_class(test_loss):
                correct_classification_counter += 1

            predicted_results.append(([copy(y.item()), copy(test_loss)]))

    classification_score = correct_classification_counter/no_samples
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    classification_score = f1
    return classification_score, predicted_results, (precision, recall, f1)
