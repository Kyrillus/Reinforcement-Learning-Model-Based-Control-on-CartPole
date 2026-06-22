"""Random shooting planner: the simplest MPC baseline.

Samples `num_sequences` random discrete action sequences of length
`horizon`, rolls each one out through the learned dynamics model, and
returns the first action of the sequence with the highest predicted
cumulative reward.
"""

from __future__ import annotations

import numpy as np

from dynamics.base import DynamicsModel, cartpole_reward_and_done


def rollout_returns(
    model: DynamicsModel,
    state: np.ndarray,
    action_sequences: np.ndarray,
) -> np.ndarray:
    """Roll out a batch of action sequences through the model and score them.

    Args:
        model: learned (or stub) dynamics model.
        state: starting state, shape (state_dim,).
        action_sequences: shape (num_sequences, horizon) of discrete actions.

    Returns:
        Predicted cumulative reward per sequence, shape (num_sequences,).
    """
    num_sequences, horizon = action_sequences.shape
    states = np.tile(state, (num_sequences, 1))
    alive = np.ones(num_sequences, dtype=bool)
    returns = np.zeros(num_sequences)

    for t in range(horizon):
        actions = action_sequences[:, t]
        next_states = model.predict(states, actions)
        rewards, done = cartpole_reward_and_done(next_states)
        # Once a rollout "dies" it stops accumulating reward, but we still
        # carry its (now meaningless) state forward to keep array shapes fixed.
        returns += np.where(alive, rewards, 0.0)
        alive &= ~done
        states = next_states

    return returns


def plan_random_shooting(
    model: DynamicsModel,
    state: np.ndarray,
    horizon: int,
    num_sequences: int,
    num_actions: int,
    rng: np.random.Generator,
) -> int:
    """Pick the first action of the best of `num_sequences` random sequences.

    Args:
        model: learned (or stub) dynamics model.
        state: current real environment state, shape (state_dim,).
        horizon: planning horizon H.
        num_sequences: number of randomly sampled action sequences.
        num_actions: size of the discrete action space.
        rng: numpy random generator for reproducibility.

    Returns:
        The action to execute next in the real environment.
    """
    action_sequences = rng.integers(0, num_actions, size=(num_sequences, horizon))
    returns = rollout_returns(model, state, action_sequences)
    best_sequence = action_sequences[np.argmax(returns)]
    return int(best_sequence[0])
