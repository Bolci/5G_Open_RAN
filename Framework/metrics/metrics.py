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
    def forward(outputs, c, nu):
        # Compute distance from center
        dist = torch.sum((outputs - c) ** 2, dim=-1)
        # SVDD loss as a combination of inliers and outliers
        return torch.mean(dist) + nu * torch.mean(torch.max(dist - torch.mean(dist), torch.zeros_like(dist)))



class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(output, x):
        recon_x, mu, log_var = output
        """Computes VAE loss: MSE reconstruction loss + KL divergence"""
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div