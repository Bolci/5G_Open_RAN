import os.path
import torch
import torch.nn as nn
from copy import copy

from Framework.postprocessors.postprocesor import Postprocessor


def test_loop(dataloader_test,
              model: nn.Module,
              loss_fn,
              postprocessor: Postprocessor,
              device="cuda"):




    test_losses_to_print = []

    with torch.no_grad():
        for X, y in dataloader_test:
            pred = model(X.to(device))
            test_loss = loss_fn(pred, X).item()
            test_losses_to_print.append(([copy(y.item()), copy(test_loss)]))

    return test_losses_to_print
