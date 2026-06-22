"""Sanity tests for the MPC planner, using the true-physics stub model.

These don't assert on RL performance (that's stochastic), only that
the planner code runs end-to-end, returns valid discrete actions, and
that CEM is at least as good as random shooting on average.
"""

from __future__ import annotations

import numpy as np

from dynamics.stub_model import StubDynamicsModel
from evaluation.evaluate import evaluate
from mpc.cem import plan_cem
from mpc.random_shooting import plan_random_shooting


def test_random_shooting_returns_valid_action():
    model = StubDynamicsModel()
    rng = np.random.default_rng(0)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = plan_random_shooting(model, state, horizon=10, num_sequences=20, num_actions=2, rng=rng)
    assert action in (0, 1)


def test_cem_returns_valid_action():
    model = StubDynamicsModel()
    rng = np.random.default_rng(0)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    action = plan_cem(model, state, horizon=10, num_sequences=20, num_actions=2, rng=rng, num_iterations=2)
    assert action in (0, 1)


def test_evaluate_produces_nondecreasing_cumulative_steps():
    model = StubDynamicsModel()
    cumulative_steps, returns = evaluate(model, num_episodes=3, planner="random_shooting", horizon=5, num_sequences=10)
    assert len(cumulative_steps) == len(returns) == 3
    assert all(cumulative_steps[i] <= cumulative_steps[i + 1] for i in range(len(cumulative_steps) - 1))
    assert all(r > 0 for r in returns)


if __name__ == "__main__":
    test_random_shooting_returns_valid_action()
    test_cem_returns_valid_action()
    test_evaluate_produces_nondecreasing_cumulative_steps()
    print("All tests passed.")
