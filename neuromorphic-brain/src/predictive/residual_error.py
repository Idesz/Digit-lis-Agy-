"""
Predictive Coding & Residual Error Driven Learning.
Hierarchical predictive model where higher layers predict incoming representation,
and only unexplained residual error (surprise) triggers synaptic plastic weight updates.
E = |Input - Prediction|
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPredictiveModule(nn.Module):
    """
    Hierarchical Autoencoding State-Space module implementing Predictive Processing.
    Computes top-down prediction of the input and evaluates the residual error vector.
    Gating mechanism ensures synaptic updates occur only when error exceeds noise threshold.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        surprise_threshold: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.surprise_threshold = surprise_threshold

        # Bottom-up recognition / encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # Top-down generative / prediction
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch_size, input_dim]
        Returns:
            prediction: [batch_size, input_dim]
            residual_error: [batch_size, input_dim] (E = |Input - Prediction|)
            latent: [batch_size, latent_dim]
        """
        latent = self.encoder(x)
        prediction = self.generator(x_latent := latent)
        residual_error = torch.abs(x - prediction)
        return prediction, residual_error, latent

    def compute_surprise_metric(self, residual_error: torch.Tensor) -> torch.Tensor:
        """
        Computes scalar mean surprise per sample: L1 norm of residual error.
        """
        return residual_error.mean(dim=-1)

    def gated_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Calculates loss gated by residual magnitude.
        Only E = |Input - Prediction| triggers substantial gradient when above threshold.
        Returns:
            loss: torch.Tensor
            mean_surprise: torch.Tensor
            is_plastic_update_needed: bool
        """
        prediction, residual_error, _ = self.forward(x)
        mean_surprise = self.compute_surprise_metric(residual_error).mean()

        # Gate synaptic weight modification based on whether surprise exceeds threshold
        is_plastic = mean_surprise.item() > self.surprise_threshold
        # Gating factor modulates gradient strength
        gate = torch.sigmoid((mean_surprise - self.surprise_threshold) * 10.0)

        # Loss is weighted by gate
        loss = gate * F.mse_loss(prediction, x)
        return loss, mean_surprise, is_plastic
