import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
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
        self.mse = nn.MSELoss(reduction='none',*args, **kwargs)

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


class SSIMLoss(nn.Module):
    """
    A custom loss function that calculates the Structural Similarity Index (SSIM) loss.

    Methods
    -------
    forward(yhat, y)
        Computes the SSIM loss between the predicted values and the actual values.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the SSIMLoss class.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        super().__init__()
        self.ssim = ssim

    def forward(self, yhat, y):
        """
        Computes the SSIM loss between the predicted values and the actual values.

        Parameters
        ----------
        yhat : torch.Tensor
            The predicted values.
        y : torch.Tensor
            The actual values.

        Returns
        -------
        torch.Tensor
            The computed SSIM loss.
        """
        data_range = yhat.max() - yhat.min()
        return 1 - self.ssim(yhat.unsqueeze(1), y.unsqueeze(1), data_range=data_range.item())

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
    def __init__(self, lambda_var=0.1):
        super().__init__()
        self.lambda_var = lambda_var


    def forward(self, output, x):
        recon_x, mu, log_var = output

        # Compute pixel-wise MSE loss
        mse_loss = (recon_x - x) ** 2
        recon_loss = mse_loss.mean()

        # Penalize the variance of reconstruction errors
        loss_variance = mse_loss.var()

        # KL Divergence
        kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div + self.lambda_var * loss_variance