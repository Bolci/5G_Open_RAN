# This script contains important loops for the development pipeline

import torch
import numpy as np
from copy import copy


def train_loop(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device)
        pred = model(X)
        loss = loss_fn(pred, X)
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


def valid_loop(dataloader, model, loss_fn, device="cuda"):
    test_losses_to_print = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            test_loss = loss_fn(pred, X).item()
            test_losses_to_print.append([copy(y), copy(test_loss)])

    test_loss_mean = np.mean(np.asarray(test_losses_to_print))
    print(f"Avg loss: {test_loss_mean:>8f} \n")

    return test_loss_mean, test_losses_to_print


def test_loop(dataloader, model, loss_fn, device="cuda"):
    test_losses_to_print = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss = loss_fn(pred, X).item()
            test_losses_to_print.append(([copy(y), copy(test_loss)]))

    return test_losses_to_print
