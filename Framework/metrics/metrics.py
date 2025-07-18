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
    def __init__(self, lambda_var=0.1, reduction='none'):
        super().__init__()
        self.lambda_var = lambda_var
        self.reduction = reduction

    def forward(self, output, x):
        recon_x, mu, log_var = output

        # Compute sample-wise MSE loss [batch_size, ...]
        mse_loss = (recon_x - x).pow(2)

        # Flatten all dimensions except batch
        if mse_loss.dim() > 2:
            mse_loss = mse_loss.view(mse_loss.size(0), -1).mean(dim=1)  # [batch_size]

        # KL Divergence per sample [batch_size]
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)

        # Total loss per sample
        loss = mse_loss + torch.mean(kl_div, dim=1)  # [batch_size]

        # Add variance penalty if needed (still computed across batch)
        if self.lambda_var > 0:
            loss_variance = mse_loss.var()  # Scalar
            loss = loss + self.lambda_var * loss_variance

        # Apply reduction if needed
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # [batch_size]