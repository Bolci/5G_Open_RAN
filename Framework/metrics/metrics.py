import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    """
    A custom loss function that calculates the Root Mean Squared Error (RMSE).

    Methods
    -------
    forward(yhat, y)
        Computes the RMSE between the predicted values and the actual values.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the RMSELoss class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__()
        self.mse = nn.MSELoss(*args, **kwargs)

    def forward(self, yhat, y):
        """
        Computes the RMSE between the predicted values and the actual values.

        Parameters
        ----------
        yhat : torch.Tensor
            The predicted values.
        y : torch.Tensor
            The actual values.

        Returns
        -------
        torch.Tensor
            The computed RMSE.
        """
        return torch.sqrt(self.mse(yhat, y))


class DSVDDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(items, truth):
        outputs, c, nu, radius = items
        # # Compute distances of outputs to the center
        # # outputs -> shape (1, 1, n_features)
        # # c -> shape (1, 1, n_features)
        # distances = torch.norm(outputs - c, dim=1)
        # # Encourage clustering of outputs close to c
        # loss = torch.mean(distances)
        # # Stronger regularization to prevent trivial outputs
        # penalty = torch.mean((distances ** 2))
        # # return loss + nu * penalty + torch.mean(outputs ** 2)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        scores = dist - radius ** 2
        loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        return loss





