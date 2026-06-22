"""Receding-horizon MPC control loop.

At every real environment step: plan an action sequence with the
chosen planner (random shooting or CEM), execute only the first
action in the real environment, then replan from the resulting state.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from dynamics.base import DynamicsModel
from mpc.cem import plan_cem
from mpc.random_shooting import plan_random_shooting

PlannerFn = Callable[[DynamicsModel, np.ndarray, int, int, int, np.random.Generator], int]

PLANNERS: dict[str, PlannerFn] = {
    "random_shooting": plan_random_shooting,
    "cem": plan_cem,
}


def run_episode(
    env: gym.Env,
    model: DynamicsModel,
    planner: str = "cem",
    horizon: int = 20,
    num_sequences: int = 200,
    rng: np.random.Generator | None = None,
    max_steps: int = 500,
) -> tuple[float, int]:
    """Run one receding-horizon MPC episode in the real environment.

    Args:
        env: real (or evaluation) Gymnasium CartPole environment.
        model: dynamics model used for planning.
        planner: "random_shooting" or "cem".
        horizon: planning horizon H.
        num_sequences: number of sampled action sequences per planning step.
        rng: numpy random generator for reproducibility.
        max_steps: safety cap on episode length.

    Returns:
        Tuple (episode_return, num_env_steps_taken).
    """
    if rng is None:
        rng = np.random.default_rng()
    plan_fn = PLANNERS[planner]
    num_actions = env.action_space.n

    state, _ = env.reset()
    episode_return = 0.0
    steps = 0

    for _ in range(max_steps):
        action = plan_fn(model, np.asarray(state), horizon, num_sequences, num_actions, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        steps += 1
        if terminated or truncated:
            break

    return episode_return, steps
