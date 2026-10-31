"""
Baseline Training and Multi-Regime Benchmarking Script for TrafficFlow AI.

Trains the Deep Q-Network agent on CPU with training seeds (1-40) using the
exact 15-D state vector, saves checkpoint to models/dqn_traffic_model.pt,
and benchmarks against Fixed-Time control across independent test seeds
(1001-1005) on Low, Medium, Rush-Hour, and Asymmetric traffic.
"""

import json
import os
import sys
import numpy as np
import pandas as pd

from agent.dqn_agent import DQNAgent
from agent.trainer import DQNTrainer
from controllers.adaptive_controller import AdaptiveDQNController
from controllers.fixed_time_controller import FixedTimeController
from environment.traffic_environment import TrafficEnvironment
from evaluation.comparison import generate_comparison_table
from evaluation.evaluator import Evaluator


def run_training_and_benchmark():
    print("=" * 65)
    print("TRAFFICFLOW AI: REFINED TRAINING & SCIENTIFIC BENCHMARK")
    print("=" * 65)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    checkpoint_path = "models/dqn_traffic_model.pt"

    # 1. Training Environment and Agent
    # Disjoint Training Seeds: 1 to 40
    env = TrafficEnvironment(
        traffic_preset="medium",
        episode_duration=180.0,
        dt=1.0,
        seed=10,
    )

    agent = DQNAgent(
        state_dim=15,
        action_dim=2,
        hidden_dim=128,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.96,
        batch_size=64,
        replay_capacity=30000,
        tau=0.01,
    )

    trainer = DQNTrainer(env, agent, checkpoint_path=checkpoint_path)

    print("\n[Phase 1] Training DQN Agent on CPU (40 episodes)...")

    def print_progress(ep, total, stats):
        if ep % 5 == 0 or ep == 1 or ep == total:
            print(
                f"Episode {ep:02d}/{total} | Reward: {stats['reward']:7.1f} | "
                f"Loss: {stats['avg_loss']:.4f} | Epsilon: {stats['epsilon']:.3f} | "
                f"Throughput: {stats['throughput']} | Avg Queue: {stats['avg_queue']:.1f}"
            )

    history = trainer.train(num_episodes=40, progress_callback=print_progress)

    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete! Saved weights to {checkpoint_path}")

    # 2. Multi-Regime Scientific Evaluation
    # Disjoint Testing Seeds: 1001, 1002, 1003, 1004, 1005
    test_seeds = [1001, 1002, 1003, 1004, 1005]
    print("\n[Phase 2] Evaluating Controllers on Pre-Generated Schedules Across 4 Regimes...")
    evaluator = Evaluator(episode_duration=240.0, dt=1.0)

    fixed_ctrl = FixedTimeController(ns_green_duration=30.0, ew_green_duration=30.0)
    dqn_ctrl = AdaptiveDQNController(agent=agent)

    regimes = ["low", "medium", "rush_hour", "asymmetric"]
    multi_regime_results = {}

    for regime in regimes:
        print(f"\nEvaluating Regime: {regime.upper()} (Seeds: {test_seeds})...")
        res = evaluator.compare_controllers(
            fixed_controller=fixed_ctrl,
            dqn_controller=dqn_ctrl,
            num_episodes=len(test_seeds),
            traffic_preset=regime,
            test_seeds=test_seeds,
        )
        comp_df = generate_comparison_table(res["fixed_metrics"], res["dqn_metrics"])
        print(comp_df.to_string(index=False))

        multi_regime_results[regime] = {
            "fixed_metrics": res["fixed_metrics"],
            "dqn_metrics": res["dqn_metrics"],
            "table": comp_df.to_dict(orient="records"),
            "sample_fixed_ts": res["sample_fixed_ts"],
            "sample_dqn_ts": res["sample_dqn_ts"],
        }

    # Save multi-regime results
    with open("results/multi_regime_results.json", "w") as f:
        json.dump(multi_regime_results, f, indent=2)

    # Save standard medium benchmark as primary results cache
    primary_data = multi_regime_results["medium"]
    with open("results/benchmark_results.json", "w") as f:
        json.dump(primary_data, f, indent=2)

    primary_df = pd.DataFrame(primary_data["table"])
    primary_df.to_csv("results/benchmark_results.csv", index=False)

    print("\n" + "=" * 65)
    print("ALL EVALUATIONS SUCCESSFULLY COMPLETED AND PERSISTED")
    print("=" * 65)


if __name__ == "__main__":
    run_training_and_benchmark()
