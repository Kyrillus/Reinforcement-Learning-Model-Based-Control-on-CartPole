"""Transition data collection from the real CartPole environment.

Provides random-policy collection (initial dataset) and helpers for
growing a dataset with transitions gathered by the MPC controller
(on-policy aggregation, as in Nagabandi et al. 2018).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

Transitions = tuple[np.ndarray, np.ndarray, np.ndarray]


def collect_random_transitions(num_steps: int, seed: int = 0) -> Transitions:
    """Collect transitions with a uniform random policy.

    Returns:
        Tuple (states, actions, next_states) with shapes
        (num_steps, 4), (num_steps,), (num_steps, 4).
    """
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    states, actions, next_states = [], [], []

    state, _ = env.reset(seed=seed)
    for _ in range(num_steps):
        action = int(rng.integers(env.action_space.n))
        next_state, _, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return np.array(states), np.array(actions), np.array(next_states)


def concat_transitions(a: Transitions, b: Transitions) -> Transitions:
    """Concatenate two transition datasets."""
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def subsample_transitions(data: Transitions, size: int, seed: int = 0) -> Transitions:
    """Randomly subsample a transition dataset to the given size."""
    states, actions, next_states = data
    idx = np.random.default_rng(seed).choice(len(states), size=size, replace=False)
    return states[idx], actions[idx], next_states[idx]
