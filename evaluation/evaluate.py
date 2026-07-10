"""Evaluation harness.

Two pieces:
  - evaluate_config: run several seeded MPC episodes for one configuration
    and report the episode returns (used by all ablations).
  - run_learning_curve: the full model-based training loop. Starting from
    random-policy data, it alternates training the dynamics model and
    collecting on-policy MPC episodes that are added back to the dataset
    (data aggregation as in Nagabandi et al. 2018). The recorded points
    give average episode return vs. real environment steps.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from data.collect import Transitions, collect_random_transitions, concat_transitions
from dynamics.base import DynamicsModel
from dynamics.nn_model import NeuralDynamicsModel
from mpc.controller import run_episode


def evaluate_config(
    model: DynamicsModel,
    planner: str = "random_shooting",
    horizon: int = 20,
    num_sequences: int = 200,
    num_episodes: int = 5,
    seed: int = 0,
) -> list[float]:
    """Run several seeded episodes for one configuration, return the returns."""
    env = gym.make("CartPole-v1")
    returns = []
    for i in range(num_episodes):
        rng = np.random.default_rng(1000 * seed + i)
        episode_return, _, _ = run_episode(
            env, model, planner=planner, horizon=horizon,
            num_sequences=num_sequences, rng=rng, reset_seed=1000 * seed + i,
        )
        returns.append(episode_return)
    env.close()
    return returns


def run_learning_curve(
    planner: str,
    seed: int = 0,
    iterations: int = 5,
    init_random_steps: int = 1000,
    episodes_per_iter: int = 3,
    horizon: int = 20,
    num_sequences: int = 200,
    model_kwargs: dict | None = None,
    fit_kwargs: dict | None = None,
) -> tuple[list[int], list[float], Transitions]:
    """Iterative model training with on-policy data aggregation.

    Each iteration trains a fresh model on all data collected so far,
    then runs MPC episodes that serve both as the evaluation of that
    model and as new training data for the next iteration.

    Returns:
        Tuple (env_steps, mean_returns, dataset): env_steps[i] is the
        number of real environment steps in the training data of the
        model evaluated at point i; dataset is the final aggregated data.
    """
    model_kwargs = model_kwargs or {}
    fit_kwargs = fit_kwargs or {}
    env = gym.make("CartPole-v1")

    dataset = collect_random_transitions(init_random_steps, seed=seed)
    env_steps: list[int] = []
    mean_returns: list[float] = []

    for it in range(iterations):
        model = NeuralDynamicsModel(seed=seed, **model_kwargs)
        model.fit(*dataset, seed=seed, **fit_kwargs)
        env_steps.append(len(dataset[0]))

        returns = []
        for ep in range(episodes_per_iter):
            episode_seed = 100_000 * seed + 100 * it + ep
            rng = np.random.default_rng(episode_seed)
            episode_return, _, transitions = run_episode(
                env, model, planner=planner, horizon=horizon,
                num_sequences=num_sequences, rng=rng, reset_seed=episode_seed,
            )
            returns.append(episode_return)
            dataset = concat_transitions(dataset, transitions)
        mean_returns.append(float(np.mean(returns)))

    env.close()
    return env_steps, mean_returns, dataset
