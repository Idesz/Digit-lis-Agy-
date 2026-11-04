import pytest
import torch
from src.predictive.residual_error import ResidualPredictiveModule


def test_residual_predictive_forward():
    batch_size = 8
    input_dim = 32
    latent_dim = 16

    module = ResidualPredictiveModule(input_dim=input_dim, latent_dim=latent_dim)
    x = torch.rand(batch_size, input_dim)

    prediction, residual_error, latent = module(x)

    assert prediction.shape == (batch_size, input_dim)
    assert residual_error.shape == (batch_size, input_dim)
    assert latent.shape == (batch_size, latent_dim)

    # Verify residual is E = |Input - Prediction|
    expected_residual = torch.abs(x - prediction)
    assert torch.allclose(residual_error, expected_residual, atol=1e-5)


def test_gated_synaptic_loss():
    input_dim = 16
    latent_dim = 8
    module = ResidualPredictiveModule(input_dim=input_dim, latent_dim=latent_dim, surprise_threshold=0.01)

    x = torch.rand(4, input_dim)
    loss, surprise, is_plastic = module.gated_loss(x)

    assert loss.item() >= 0.0
    assert surprise.item() >= 0.0
    assert isinstance(is_plastic, bool)

    loss.backward()
    for param in module.parameters():
        assert param.grad is not None
