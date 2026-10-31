"""
Training Pipeline for Deep Q-Network Traffic Agent.

Manages multi-episode training loops, real gradient updates, performance logging,
and model checkpoint persistence with optional callback hooks for UI integration.
"""

from typing import Callable, Dict, List, Optional
import numpy as np

from agent.dqn_agent import DQNAgent
from environment.traffic_environment import TrafficEnvironment


class DQNTrainer:
    """Manages the end-to-end DQN training lifecycle."""

    def __init__(
        self,
        env: TrafficEnvironment,
        agent: DQNAgent,
        checkpoint_path: str = "models/dqn_traffic_model.pt",
    ):
        """
        Initialize the trainer.

        Args:
            env: Traffic simulation environment.
            agent: Deep Q-Network agent instance.
            checkpoint_path: Filepath where model weights will be saved.
        """
        self.env = env
        self.agent = agent
        self.checkpoint_path = checkpoint_path

        # Training history logs
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_epsilons: List[float] = []
        self.episode_throughputs: List[int] = []
        self.episode_avg_queues: List[float] = []
        self.best_reward = -float("inf")

    def train(
        self,
        num_episodes: int = 50,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, List[float]]:
        """
        Execute training across specified number of episodes.

        Args:
            num_episodes: Total number of training episodes.
            progress_callback: Optional callback func(current_ep, total_ep, ep_stats).
            stop_flag: Optional callable returning True if user requested early stop.

        Returns:
            Dictionary containing training history metrics.
        """
        for episode in range(1, num_episodes + 1):
            if stop_flag is not None and stop_flag():
                print(f"Training stopped early at episode {episode - 1}.")
                break

            state = self.env.reset(seed=episode * 17)
            done = False
            total_reward = 0.0
            episode_losses = []

            while not done:
                action = self.agent.select_action(state, eval_mode=False)
                next_state, reward, done, info = self.env.step(action)

                # Execute Bellman learning step on replay buffer
                loss = self.agent.step(state, action, reward, next_state, done)
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

            # End of episode updates
            avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
            current_eps = self.agent.decay_epsilon()
            total_throughput = len(self.env.completed_vehicles)
            avg_queue = float(np.mean(self.env.history_total_queue)) if self.env.history_total_queue else 0.0

            self.episode_rewards.append(total_reward)
            self.episode_losses.append(avg_loss)
            self.episode_epsilons.append(current_eps)
            self.episode_throughputs.append(total_throughput)
            self.episode_avg_queues.append(avg_queue)

            # Save best checkpoint
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save(self.checkpoint_path)

            ep_stats = {
                "episode": episode,
                "reward": round(total_reward, 2),
                "avg_loss": round(avg_loss, 4),
                "epsilon": round(current_eps, 4),
                "throughput": total_throughput,
                "avg_queue": round(avg_queue, 2),
            }

            if progress_callback is not None:
                progress_callback(episode, num_episodes, ep_stats)

        # Always save the latest model as well
        self.agent.save(self.checkpoint_path)

        return {
            "rewards": self.episode_rewards,
            "losses": self.episode_losses,
            "epsilons": self.episode_epsilons,
            "throughputs": self.episode_throughputs,
            "avg_queues": self.episode_avg_queues,
        }
