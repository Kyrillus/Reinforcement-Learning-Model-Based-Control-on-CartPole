"""Evaluation harness: episode return vs. environment steps.

This is the primary learning-curve metric required by the project
sheet (section 2d). It runs repeated receding-horizon MPC episodes,
tracks cumulative environment steps against episode return, and plots
the resulting learning curve.

Run directly to produce `evaluation/learning_curve.png` using the
stub (true-physics) dynamics model. Swap in the real trained model
once it's available (see `dynamics/base.py`).
"""

from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dynamics.base import DynamicsModel
from dynamics.stub_model import StubDynamicsModel
from mpc.controller import run_episode


def evaluate(
    model: DynamicsModel,
    num_episodes: int = 20,
    planner: str = "cem",
    horizon: int = 20,
    num_sequences: int = 200,
    seed: int = 0,
) -> tuple[list[int], list[float]]:
    """Run several MPC episodes and record (cumulative_steps, return) pairs.

    Args:
        model: dynamics model used for planning.
        num_episodes: number of evaluation episodes to run.
        planner: "random_shooting" or "cem".
        horizon: planning horizon H.
        num_sequences: number of sampled action sequences per planning step.
        seed: seed for reproducibility.

    Returns:
        Tuple (cumulative_steps, returns) — parallel lists, one entry per episode.
    """
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)

    cumulative_steps = []
    returns = []
    total_steps = 0

    for _ in range(num_episodes):
        episode_return, steps = run_episode(
            env, model, planner=planner, horizon=horizon,
            num_sequences=num_sequences, rng=rng,
        )
        total_steps += steps
        cumulative_steps.append(total_steps)
        returns.append(episode_return)

    env.close()
    return cumulative_steps, returns


def plot_learning_curve(
    cumulative_steps: list[int],
    returns: list[float],
    save_path: str = "evaluation/learning_curve.png",
) -> None:
    """Plot and save episode return vs. environment steps."""
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_steps, returns, marker="o")
    plt.xlabel("Environment steps")
    plt.ylabel("Episode return")
    plt.title("MPC learning curve on CartPole-v1")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Demo run with the stub (true-physics) model so the harness is
    # testable before the real learned dynamics model exists.
    stub_model = StubDynamicsModel()
    steps, ep_returns = evaluate(stub_model, num_episodes=10)
    plot_learning_curve(steps, ep_returns)
    print("Episode returns:", ep_returns)
