# This script contains important loops for the development pipeline

import torch
from copy import copy
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(res, dim=(-1, -2, -3))

koef = 8
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
    model.output_attention = True
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X_ = X.to(device)
        if model.model_name == "anomaly_transformer":
            pred, series, prior, _ = model(X_)
            series_loss = 0.0
            prior_loss = 0.0
            recon_loss = loss_fn(pred, X_)
            recon_loss = recon_loss.mean()
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.sum(prior[u], dim=-1, keepdim=True)

                series_loss += torch.mean(my_kl_loss(series[u], norm_prior.detach()) + my_kl_loss(norm_prior.detach(), series[u]))
                prior_loss += torch.mean(my_kl_loss(norm_prior, series[u].detach()) + my_kl_loss(series[u].detach(), norm_prior))

            series_loss /= len(prior)
            prior_loss /= len(prior)
            # koef = 0.7
            loss1 = recon_loss - koef * series_loss
            loss2 = recon_loss + koef * prior_loss
            loss = loss1 + loss2
        else:
            pred = model(X_)
            loss = loss_fn(pred, X_).mean()
        # Backpropagation
        optimizer.zero_grad()
        if model.model_name == "anomaly_transformer":

            loss1 = loss - koef * series_loss
            loss2 = loss + koef * prior_loss
            loss1.backward(retain_graph=True)
            loss2.backward()
            loss = loss1 + loss2
        else:
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
        A tuple containing:
        - float: The average validation loss.
        - np.ndarray: Losses to print (if not during training).
        - np.ndarray: Per-sample validation losses.
    """
    val_losses_to_print = []
    val_losses_score = []
    model.output_attention = True
    with torch.no_grad():
        for X, y in dataloader:
            X_ = X.to(device)
            y = y.to(device)
            if model.model_name == "anomaly_transformer":

                pred, series, prior, _ = model(X_)

                # Normalized prior
                normalized_prior = []
                for u in range(len(prior)):
                    norm = torch.sum(prior[u], dim=-1, keepdim=True)
                    norm_prior = prior[u] / norm.repeat(1, 1, 1, model.win_size)
                    normalized_prior.append(norm_prior)

                # Series-Prior KL losses
                series_loss = 0.0
                prior_loss = 0.0
                # koef = 0.7
                for u in range(len(prior)):
                    s = series[u]
                    p = normalized_prior[u]
                    series_loss += torch.mean(my_kl_loss(s, p.detach()) + my_kl_loss(p.detach(), s))
                    prior_loss += torch.mean(my_kl_loss(p, s.detach()) + my_kl_loss(s.detach(), p))
                series_loss /= len(prior)
                prior_loss /= len(prior)

                # Final loss
                recon_loss = loss_fn(pred, X_).mean(dim=(1,2))
                val_loss = recon_loss + koef * prior_loss - koef * series_loss
            else:
                pred = model(X_)
                val_loss = loss_fn(pred, X_).mean(dim=(1,2))

            val_losses_score.append(copy(val_loss))

            if not is_train:
                val_losses_to_print.append(torch.vstack([copy(y), copy(val_loss)]))

    val_losses_to_print = torch.concatenate(val_losses_to_print, dim=1).cpu().numpy().T if val_losses_to_print else []
    val_losses_score = torch.concatenate(val_losses_score, dim=0)
    val_loss_mean = torch.mean(val_losses_score).item()
    val_losses_score = val_losses_score.cpu().numpy()
    print(f"Avg loss: {val_loss_mean:>8f} \n")


    # val_loss_mean - FLOAT - average loss over all samples in the validation set
    # val_losses_to_print - np.ndarray - array of shape [BATCH, 2] with first column being labels and second column being losses
    # val_losses_score - np.ndarray - array of shape [BATCH, ] with per-sample losses
    return val_loss_mean, val_losses_to_print, val_losses_score

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
    model.output_attention = True
    with torch.no_grad():
        for X, y in dataloader_test:
            if not (len(X.shape) == 3):
                X = X.unsqueeze(dim=0)
            X_ = X.to(device)
            y = y.to(device)
            # ----
            if model.model_name == "anomaly_transformer":

                pred, series, prior, _ = model(X_)

                # Normalized prior
                normalized_prior = []
                for u in range(len(prior)):
                    norm = torch.sum(prior[u], dim=-1, keepdim=True)
                    norm_prior = prior[u] / norm.repeat(1, 1, 1, model.win_size)
                    normalized_prior.append(norm_prior)

                # Series-Prior KL losses
                series_loss_all = []
                prior_loss_all = []

                for u in range(len(prior)):
                    s = series[u]  # [B, H, N, W]
                    p = normalized_prior[u]  # [B, H, N, W]
                    # Compute KL divergence for each sample in the batch
                    kl_sp = (my_kl_loss(s, p.detach()) + my_kl_loss(p.detach(), s))
                    kl_ps = (my_kl_loss(p, s.detach()) + my_kl_loss(s.detach(), p))

                    series_loss_all.append(kl_sp)  # list of [B]
                    prior_loss_all.append(kl_ps)  # list of [B]

                # Average across the attention heads (u) but keep per-batch info
                series_loss = torch.stack(series_loss_all).mean(dim=0)  # [B]
                prior_loss = torch.stack(prior_loss_all).mean(dim=0)  # [B]

                # Final loss
                recon_loss = loss_fn(pred, X_).mean(dim=(1,2))

                test_loss = recon_loss + koef * prior_loss - koef * series_loss
            else:
                pred = model(X_)
                test_loss = loss_fn(pred, X_).mean(dim=(1,2))

            # total_loss = total_loss.item()
            predicted_label = predict_class(test_loss)
            predicted_labels.append(predicted_label)
            true_labels.append(y)

            # if y == predict_class(total_loss):
            #     correct_classification_counter += 1

            predicted_results.append(torch.vstack([copy(y), copy(test_loss)]))
    true_labels = torch.concatenate(true_labels, dim=0).cpu().numpy()
    predicted_labels = torch.concatenate(predicted_labels, dim=0).cpu().numpy()
    # classification_score = correct_classification_counter/no_samples
    precision = precision_score(true_labels, predicted_labels, zero_division=0.0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0.0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0.0)

    classification_score = f1
    predicted_results = torch.concatenate(predicted_results, dim=1).cpu().numpy()
    return classification_score, predicted_results, (precision, recall, f1)
