"""Vectorized true CartPole-v1 dynamics.

Implements the exact update equations of Gymnasium's CartPole-v1 (Euler
integration) as a batched numpy computation. Used as an oracle baseline
for the MPC planner (planning with perfect dynamics) and in the tests,
where it replaces stepping a real environment sample by sample.
"""

from __future__ import annotations

import numpy as np

from dynamics.base import DynamicsModel

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # half the pole length
POLE_MASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02


class TrueDynamicsModel(DynamicsModel):
    """Batched analytic CartPole physics behind the DynamicsModel interface."""

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = states.T
        force = np.where(actions == 1, FORCE_MAG, -FORCE_MAG)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        temp = (force + POLE_MASS_LENGTH * theta_dot**2 * sin_theta) / TOTAL_MASS
        theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
            LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta**2 / TOTAL_MASS)
        )
        x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

        return np.stack(
            [
                x + TAU * x_dot,
                x_dot + TAU * x_acc,
                theta + TAU * theta_dot,
                theta_dot + TAU * theta_acc,
            ],
            axis=1,
        )
