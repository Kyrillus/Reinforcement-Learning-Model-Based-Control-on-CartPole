"""Temporary stand-in dynamics model used to develop and test the MPC planner
before learned neural dynamics model is ready.

This wraps the *true* CartPole physics (via a throwaway Gymnasium
environment) behind the same `DynamicsModel` interface, so the planner
code in `mpc/` can be written and tested independently of the model
training work. Replace `StubDynamicsModel` with the trained neural
model for the final submission.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from dynamics.base import DynamicsModel


class StubDynamicsModel(DynamicsModel):
    """Predicts next states by stepping a private copy of CartPole-v1."""

    def __init__(self) -> None:
        self._env = gym.make("CartPole-v1").unwrapped

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        next_states = np.empty_like(states)
        for i, (state, action) in enumerate(zip(states, actions)):
            self._env.state = tuple(state.tolist())
            next_state, _, _, _, _ = self._env.step(int(action))
            next_states[i] = next_state
        return next_states
