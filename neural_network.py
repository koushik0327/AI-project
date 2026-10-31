"""
Deep Q-Network PyTorch Neural Network Architecture.

Implements a multi-layer perceptron with Layer Normalization and ReLU activations
to estimate Q-values Q(s, a) for traffic signal phase actions.
"""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Deep Q-Network approximating the optimal action-value function Q*(s, a)."""

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 2,
        hidden_dim: int = 128,
    ):
        """
        Initialize the network.

        Args:
            state_dim: Size of the normalized input state vector (default 15).
            action_dim: Number of discrete signal phase actions (default 2).
            hidden_dim: Number of neurons in hidden layers (default 128).
        """
        super(DQNNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Initialize network weights using He/Kaiming initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming uniform weight initialization for stable gradients."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting Q-values for each action.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).

        Returns:
            Tensor of shape (batch_size, action_dim) containing Q-values.
        """
        return self.net(state)
