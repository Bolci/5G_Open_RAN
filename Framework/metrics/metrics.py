import torch.nn as nn
import torch
import torch.nn.functional as F

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

class MSEwithVarianceLoss(nn.Module):
    """
    A custom loss function that calculates the Root Mean Squared Error (RMSE).

    Methods
    -------
    forward(yhat, y)
        Computes the RMSE between the predicted values and the actual values.
    """

    def __init__(self, lambda_var=0.1):
        super().__init__()
        self.lambda_var = lambda_var

    def forward(self, yhat, y):
        # Compute pixel-wise MSE loss
        mse_loss = (yhat - y) ** 2
        recon_loss = mse_loss.mean()
        # Penalize the variance of reconstruction errors
        loss_variance = mse_loss.var()
        return recon_loss + self.lambda_var * loss_variance


class unified_loss_fn(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(output, target, lambda_reg=1e-3):
        """
        Unified loss function that adapts to different model outputs.

        For models that output a single tensor (e.g., a standard autoencoder),
        it returns the mean squared reconstruction loss.

        For models that output a tuple of 5 elements (assumed to be:
        (x_recon, x_cf, disc_real, disc_fake, delta) as in the adversarial
        counterfactual reconstruction module), the loss is computed as follows:
          - In training mode (i.e. when gradients are enabled), the loss is:
                total_loss = rec_loss + adv_loss + reg_loss
            where:
                rec_loss = MSE(x_recon, target)
                adv_loss = BCE(disc_fake, ones)
                reg_loss = lambda_reg * mean(|delta|)
          - In evaluation mode (i.e. inside a torch.no_grad() block),
            only the reconstruction loss (rec_loss) is returned.

        Parameters
        ----------
        output : torch.Tensor or tuple
            The output from the model. Either a single tensor or a tuple
            (x_recon, x_cf, disc_real, disc_fake, delta).
        target : torch.Tensor
            The target tensor (typically the input data for reconstruction).
        lambda_reg : float, optional
            Regularization strength for the perturbation term (default is 1e-3).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        # If output is a tuple of 5 elements, assume it comes from the adversarial model.
        if isinstance(output, tuple) and len(output) == 5:
            x_recon, x_cf, disc_real, disc_fake, delta = output
            # Compute standard reconstruction loss.
            rec_loss = F.mse_loss(x_recon, target, reduction='mean')

            # If gradients are enabled (training mode), add adversarial and regularization losses.
            if torch.is_grad_enabled():
                valid = torch.ones_like(disc_real)
                # Only use adversarial loss for the generator: encourage disc_fake to be classified as real.
                adv_loss = F.binary_cross_entropy(disc_fake, valid)
                reg_loss = lambda_reg * torch.mean(torch.abs(delta))
                total_loss = rec_loss + adv_loss + reg_loss
                return total_loss
            else:
                # In evaluation mode, only return the reconstruction error (used for thresholding).
                return rec_loss
        else:
            # For standard models that return a single tensor, just compute the reconstruction loss.
            return F.mse_loss(output, target, reduction='mean')
