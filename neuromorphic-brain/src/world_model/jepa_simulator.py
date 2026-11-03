"""
World Model & Virtual Rollout Simulation (JEPA Architecture).
Simulates future environmental transitions and action evaluations in an abstract latent embedding space
rather than computationally expensive pixel/signal space.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPASimulator(nn.Module):
    """
    Joint-Embedding Predictive Architecture (JEPA) simulator.
    Predicts next latent representation given current latent state and action/perturbation:
    s_{t+1} = T(s_t, a_t)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Predictor network in latent space
        self.transition_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Evaluator / Value estimation head (predicts outcome expected reward or stability)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def predict_next_latent(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Single transition step in latent space.
        latent_state: [batch_size, latent_dim]
        action: [batch_size, action_dim]
        Returns:
            next_latent: [batch_size, latent_dim] (residual connection for stability)
        """
        combined = torch.cat([latent_state, action], dim=-1)
        delta = self.transition_net(combined)
        next_latent = latent_state + delta
        return next_latent

    def simulate_rollout(
        self,
        initial_latent: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Executes multi-step virtual rollout in latent space.
        initial_latent: [batch_size, latent_dim]
        action_sequence: [batch_size, horizon, action_dim]
        Returns:
            trajectory: list of predicted latent states across horizon
            total_value: accumulated evaluated value across trajectory [batch_size, 1]
        """
        batch_size, horizon, _ = action_sequence.shape
        current_state = initial_latent
        trajectory = []
        total_value = torch.zeros(batch_size, 1, device=initial_latent.device)

        for step in range(horizon):
            action_t = action_sequence[:, step, :]
            current_state = self.predict_next_latent(current_state, action_t)
            trajectory.append(current_state)
            step_val = self.value_head(current_state)
            total_value = total_value + step_val

        return trajectory, total_value
