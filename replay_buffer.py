"""
Experience Replay Buffer for Deep Q-Network Agent.

Stores observed transition tuples (state, action, reward, next_state, done)
and supplies decorrelated uniform random mini-batches for Bellman updates.
"""

from collections import deque
import random
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity experience replay memory buffer."""

    def __init__(self, capacity: int = 50000):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new transition tuple to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions and convert to PyTorch tensors.

        Args:
            batch_size: Number of transitions to sample.
            device: Target torch device ('cpu' or 'cuda').

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        """Return the current number of transitions stored."""
        return len(self.buffer)
