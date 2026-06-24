"""Interface that any dynamics model must implement to be used by the MPC planner.

This file only defines the contract the planner relies on, plus the
CartPole-specific reward/termination logic needed for planning, so the
two halves of the project can be developed independently.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

# CartPole-v1 termination thresholds (must match the Gymnasium environment).
CARTPOLE_X_THRESHOLD = 2.4
CARTPOLE_THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360


class DynamicsModel(ABC):
    """Abstract predictive model f_theta(s_t, a_t) -> s_{t+1}.

    Implementations may predict the next state directly or a state
    delta internally, as long as `predict` returns the next state.
    """

    @abstractmethod
    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict next states for a batch of (state, action) pairs.

        Args:
            states: array of shape (N, state_dim).
            actions: array of shape (N,) with discrete action indices.

        Returns:
            Array of shape (N, state_dim) with predicted next states.
        """
        raise NotImplementedError


def cartpole_reward_and_done(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CartPole-v1 reward and termination check from raw states.

    CartPole-v1 gives a reward of +1 for every step the pole has not
    fallen and the cart has not left the track. This lets the planner
    score imagined rollouts without needing access to the real environment.

    Args:
        states: array of shape (N, 4) with columns
            [cart_position, cart_velocity, pole_angle, pole_angular_velocity].

    Returns:
        Tuple (rewards, dones), each of shape (N,).
    """
    x = states[:, 0]
    theta = states[:, 2]
    done = (
        (x < -CARTPOLE_X_THRESHOLD)
        | (x > CARTPOLE_X_THRESHOLD)
        | (theta < -CARTPOLE_THETA_THRESHOLD_RADIANS)
        | (theta > CARTPOLE_THETA_THRESHOLD_RADIANS)
    )
    reward = np.where(done, 0.0, 1.0)
    return reward, done
