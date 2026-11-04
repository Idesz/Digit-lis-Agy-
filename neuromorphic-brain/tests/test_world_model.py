import pytest
import torch
from src.world_model.jepa_simulator import JEPASimulator


def test_jepa_single_step():
    batch_size = 4
    latent_dim = 16
    action_dim = 4

    jepa = JEPASimulator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32)
    s_t = torch.randn(batch_size, latent_dim)
    a_t = torch.randn(batch_size, action_dim)

    s_next = jepa.predict_next_latent(s_t, a_t)
    assert s_next.shape == (batch_size, latent_dim)


def test_jepa_virtual_rollout():
    batch_size = 2
    latent_dim = 16
    action_dim = 4
    horizon = 5

    jepa = JEPASimulator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=32)
    s_0 = torch.randn(batch_size, latent_dim)
    action_seq = torch.randn(batch_size, horizon, action_dim)

    trajectory, total_value = jepa.simulate_rollout(s_0, action_seq)

    assert len(trajectory) == horizon
    for state in trajectory:
        assert state.shape == (batch_size, latent_dim)
    assert total_value.shape == (batch_size, 1)

    loss = total_value.sum()
    loss.backward()
    for param in jepa.parameters():
        assert param.grad is not None
