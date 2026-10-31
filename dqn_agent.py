"""
Deep Q-Network (DQN) Agent for Traffic Signal Control.

Implements the DQN algorithm from scratch in PyTorch, including:
- Epsilon-greedy exploration policy
- Experience replay buffer interaction
- Bellman equation target calculation: y_i = r + gamma * max_a' Q_target(s', a')
- Huber / Smooth L1 loss backpropagation
- Target network soft/hard synchronization
- Model checkpoint saving and loading
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.neural_network import DQNNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """Deep Q-Network Reinforcement Learning Agent."""

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 2,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 50000,
        tau: float = 0.01,
        device: Optional[str] = None,
    ):
        """
        Initialize the DQN Agent.

        Args:
            state_dim: Size of state vector (15).
            action_dim: Number of actions (2: NS_GREEN, EW_GREEN).
            hidden_dim: Number of hidden units in network.
            learning_rate: Optimizer learning rate for Adam.
            gamma: Future discount factor gamma in [0, 1].
            epsilon_start: Initial exploration rate.
            epsilon_end: Minimum exploration rate.
            epsilon_decay: Multiplicative decay factor per episode or step.
            batch_size: Mini-batch size for gradient updates.
            replay_capacity: Max transitions stored in replay buffer.
            tau: Target network soft-update parameter (tau << 1.0).
            device: Torch device ('cpu' or 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        # Epsilon exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Set computation device (default CPU for university laptop compatibility)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural Networks: Policy Net (trained) and Target Net (frozen targets)
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for evaluation/inference

        # Optimizer and Huber loss for robust gradients
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss reduces sensitivity to outliers

        # Experience Replay Buffer
        self.memory = ReplayBuffer(capacity=replay_capacity)

        # Diagnostics
        self.total_steps = 0
        self.recent_losses = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select an action using epsilon-greedy exploration.

        Args:
            state: 15-D state vector.
            eval_mode: If True, disables exploration and acts purely greedily.

        Returns:
            Selected action (0: NS_GREEN, 1: EW_GREEN).
        """
        if not eval_mode and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        # Greedy action from policy network
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        self.policy_net.train()
        return action

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Record transition in replay buffer and perform a learning update if ready.

        Returns:
            Loss value if learning occurred, else None.
        """
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

        # Train if enough samples exist in memory
        loss = None
        if len(self.memory) >= self.batch_size:
            loss = self.learn()

        return loss

    def learn(self) -> float:
        """
        Perform one gradient descent step using a mini-batch from the replay buffer.
        Applies the Bellman Optimality Equation:
            y = r + (1 - done) * gamma * max_a' Q_target(s', a')

        Returns:
            Computed scalar loss.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, device=self.device
        )

        # Predicted Q-values: Q(s, a)
        q_eval = self.policy_net(states).gather(1, actions)

        # Target Q-values using Target Network: max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Bellman equation
            q_target = rewards + (1.0 - dones) * (self.gamma * q_next)

        # Compute loss
        loss = self.criterion(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Target network soft update: theta_target = tau * theta_policy + (1 - tau) * theta_target
        self._soft_update_target_network()

        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        return loss_val

    def _soft_update_target_network(self) -> None:
        """Softly synchronize target network weights with policy network."""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def decay_epsilon(self) -> float:
        """Decay exploration rate after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, filepath: str) -> None:
        """
        Save agent weights and training hyperparameters to disk.

        Args:
            filepath: Destination file path (e.g., 'models/dqn_agent.pt').
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> bool:
        """
        Load agent weights from disk.

        Args:
            filepath: Source file path.

        Returns:
            True if load succeeded, False otherwise.
        """
        if not os.path.exists(filepath):
            return False

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            self.total_steps = checkpoint.get("total_steps", 0)
            return True
        except Exception as e:
            print(f"Error loading checkpoint from {filepath}: {e}")
            return False
