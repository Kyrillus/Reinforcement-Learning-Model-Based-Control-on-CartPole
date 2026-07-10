"""Sanity tests: planners run end-to-end, the true-dynamics model matches
the real environment, and the neural model trains and predicts sensibly."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from data.collect import collect_random_transitions
from dynamics.nn_model import NeuralDynamicsModel
from dynamics.true_model import TrueDynamicsModel
from evaluation.evaluate import evaluate_config
from mpc.cem import plan_cem
from mpc.random_shooting import plan_random_shooting


def test_true_model_matches_gymnasium():
    env = gym.make("CartPole-v1").unwrapped
    model = TrueDynamicsModel()
    rng = np.random.default_rng(0)
    state, _ = env.reset(seed=0)
    for _ in range(50):
        action = int(rng.integers(2))
        predicted = model.predict(state[None], np.array([action]))[0]
        state, _, terminated, truncated, _ = env.step(action)
        np.testing.assert_allclose(predicted, state, atol=1e-6)
        if terminated or truncated:
            state, _ = env.reset()


def test_planners_return_valid_actions():
    model = TrueDynamicsModel()
    rng = np.random.default_rng(0)
    state = np.array([0.0, 0.0, 0.0, 0.0])
    for plan_fn in (plan_random_shooting, plan_cem):
        action = plan_fn(model, state, horizon=10, num_sequences=20, num_actions=2, rng=rng)
        assert action in (0, 1)


def test_mpc_with_true_dynamics_balances():
    model = TrueDynamicsModel()
    returns = evaluate_config(model, planner="random_shooting", horizon=20,
                              num_sequences=100, num_episodes=2, seed=0)
    assert all(r >= 100 for r in returns)


def test_neural_model_fits_and_predicts():
    data = collect_random_transitions(500, seed=0)
    for predict_delta in (True, False):
        model = NeuralDynamicsModel(hidden_size=32, predict_delta=predict_delta)
        history = model.fit(*data, epochs=30)
        assert history["best_val_loss"] < 0.5
        preds = model.predict(data[0][:10], data[1][:10])
        assert preds.shape == (10, 4)
        assert np.isfinite(preds).all()
