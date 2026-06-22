"""Cross-Entropy Method (CEM) planner over discrete action sequences.

CEM refines random shooting by iteratively re-fitting a per-timestep
categorical distribution over actions to the top-performing (elite)
sampled sequences, instead of sampling uniformly at random every time.
"""

from __future__ import annotations

import numpy as np

from dynamics.base import DynamicsModel
from mpc.random_shooting import rollout_returns


def plan_cem(
    model: DynamicsModel,
    state: np.ndarray,
    horizon: int,
    num_sequences: int,
    num_actions: int,
    rng: np.random.Generator,
    num_iterations: int = 5,
    elite_frac: float = 0.1,
) -> int:
    """Pick the first action found by iteratively refining a CEM distribution.

    Args:
        model: learned (or stub) dynamics model.
        state: current real environment state, shape (state_dim,).
        horizon: planning horizon H.
        num_sequences: number of action sequences sampled per CEM iteration.
        num_actions: size of the discrete action space.
        rng: numpy random generator for reproducibility.
        num_iterations: number of CEM refinement rounds.
        elite_frac: fraction of top sequences used to refit the distribution.

    Returns:
        The action to execute next in the real environment.
    """
    num_elite = max(1, int(num_sequences * elite_frac))

    # Per-timestep categorical distribution over actions, initialized uniform.
    probs = np.full((horizon, num_actions), 1.0 / num_actions)

    best_action = 0
    for _ in range(num_iterations):
        action_sequences = np.empty((num_sequences, horizon), dtype=int)
        for t in range(horizon):
            action_sequences[:, t] = rng.choice(
                num_actions, size=num_sequences, p=probs[t]
            )

        returns = rollout_returns(model, state, action_sequences)
        elite_idx = np.argsort(returns)[-num_elite:]
        elite_sequences = action_sequences[elite_idx]

        # Refit the categorical distribution to the elite sequences.
        for t in range(horizon):
            counts = np.bincount(elite_sequences[:, t], minlength=num_actions)
            probs[t] = counts / counts.sum()

        best_action = int(elite_sequences[-1, 0])

    return best_action
