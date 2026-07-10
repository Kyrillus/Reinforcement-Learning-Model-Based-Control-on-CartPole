"""Ablation experiments over the MPC planner and the dynamics model.

Covers:
  - planning horizon H,
  - number of sampled action sequences N,
  - random shooting vs. CEM,
  - predicting the state delta vs. the absolute next state,
  - dynamics-model size and training data amount,
plus baselines (random policy, MPC with the true dynamics as an oracle)
and open-loop multi-step prediction error for the model variants.

All planner ablations share one dynamics model trained on a fixed
dataset, so differences come from the planner alone. Model ablations
retrain on the same dataset (or subsets of it) and evaluate with the
default planner settings.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from data.collect import (
    Transitions,
    collect_random_transitions,
    concat_transitions,
    subsample_transitions,
)
from dynamics.base import DynamicsModel
from dynamics.nn_model import NeuralDynamicsModel
from dynamics.true_model import TrueDynamicsModel
from evaluation.evaluate import evaluate_config
from mpc.controller import run_episode

DEFAULT_HORIZON = 20
DEFAULT_NUM_SEQUENCES = 200
PLANNERS = ("random_shooting", "cem")


def build_ablation_dataset(
    target_size: int = 8000,
    init_random_steps: int = 2000,
    episodes_per_round: int = 3,
    seed: int = 0,
) -> Transitions:
    """Collect a fixed dataset for the ablations: random data plus MPC data.

    Starts from random-policy transitions and grows the dataset with
    episodes from an MPC controller that is retrained between rounds,
    so the data also covers the states a competent controller visits.
    """
    env = gym.make("CartPole-v1")
    dataset = collect_random_transitions(init_random_steps, seed=seed)
    round_idx = 0
    while len(dataset[0]) < target_size:
        model, _ = train_model(dataset, seed=seed)
        for ep in range(episodes_per_round):
            episode_seed = 500_000 + 100 * round_idx + ep + seed
            rng = np.random.default_rng(episode_seed)
            _, _, transitions = run_episode(
                env, model, planner="random_shooting", horizon=DEFAULT_HORIZON,
                num_sequences=DEFAULT_NUM_SEQUENCES, rng=rng, reset_seed=episode_seed,
            )
            dataset = concat_transitions(dataset, transitions)
        round_idx += 1

    env.close()
    return subsample_transitions(dataset, target_size, seed=seed)


def summarize(returns: list[float]) -> dict:
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "returns": returns,
    }


def train_model(dataset: Transitions, seed: int = 0, **model_kwargs) -> tuple[NeuralDynamicsModel, dict]:
    """Train a dynamics model on the dataset, return it with its fit history."""
    model = NeuralDynamicsModel(seed=seed, **model_kwargs)
    history = model.fit(*dataset, seed=seed)
    return model, history


def ablation_horizon(
    model: DynamicsModel, horizons: list[int], num_episodes: int, seed: int
) -> dict:
    results: dict = {}
    for planner in PLANNERS:
        results[planner] = {}
        for horizon in horizons:
            returns = evaluate_config(
                model, planner=planner, horizon=horizon,
                num_sequences=DEFAULT_NUM_SEQUENCES,
                num_episodes=num_episodes, seed=seed,
            )
            results[planner][horizon] = summarize(returns)
    return results


def ablation_num_sequences(
    model: DynamicsModel, sequence_counts: list[int], num_episodes: int, seed: int
) -> dict:
    results: dict = {}
    for planner in PLANNERS:
        results[planner] = {}
        for num_sequences in sequence_counts:
            returns = evaluate_config(
                model, planner=planner, horizon=DEFAULT_HORIZON,
                num_sequences=num_sequences,
                num_episodes=num_episodes, seed=seed,
            )
            results[planner][num_sequences] = summarize(returns)
    return results


def random_policy_returns(num_episodes: int, seed: int) -> list[float]:
    """Episode returns of a uniformly random policy."""
    env = gym.make("CartPole-v1")
    returns = []
    for i in range(num_episodes):
        rng = np.random.default_rng(seed + i)
        _, _ = env.reset(seed=seed + i)
        episode_return, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(2)))
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
    env.close()
    return returns


def baselines(model: DynamicsModel, num_episodes: int, seed: int) -> dict:
    """Random policy, MPC with the learned model, and MPC with true dynamics."""
    results: dict = {}
    results["random_policy"] = summarize(random_policy_returns(num_episodes, seed))

    oracle = TrueDynamicsModel()
    for planner in PLANNERS:
        results[f"learned_{planner}"] = summarize(evaluate_config(
            model, planner=planner, horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, num_episodes=num_episodes, seed=seed,
        ))
        results[f"oracle_{planner}"] = summarize(evaluate_config(
            oracle, planner=planner, horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, num_episodes=num_episodes, seed=seed,
        ))
    return results


def ablation_delta_vs_absolute(
    dataset: Transitions, num_episodes: int, seed: int
) -> tuple[dict, dict[str, NeuralDynamicsModel]]:
    """Train delta and absolute variants on the same data, compare returns."""
    results: dict = {}
    models: dict[str, NeuralDynamicsModel] = {}
    for name, predict_delta in (("delta", True), ("absolute", False)):
        model, history = train_model(dataset, seed=seed, predict_delta=predict_delta)
        returns = evaluate_config(
            model, planner="random_shooting", horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, num_episodes=num_episodes, seed=seed,
        )
        results[name] = summarize(returns)
        results[name]["val_mse"] = history["best_val_loss"]
        models[name] = model
    return results, models


def ablation_model_size(
    dataset: Transitions, hidden_sizes: list[int], num_episodes: int, seed: int
) -> dict:
    results: dict = {}
    for hidden_size in hidden_sizes:
        model, history = train_model(dataset, seed=seed, hidden_size=hidden_size)
        returns = evaluate_config(
            model, planner="random_shooting", horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, num_episodes=num_episodes, seed=seed,
        )
        results[hidden_size] = summarize(returns)
        results[hidden_size]["val_mse"] = history["best_val_loss"]
        results[hidden_size]["num_parameters"] = model.num_parameters()
    return results


def ablation_data_amount(
    dataset: Transitions, data_sizes: list[int], num_episodes: int, seed: int
) -> dict:
    results: dict = {}
    for size in data_sizes:
        subset = subsample_transitions(dataset, size, seed=seed)
        model, history = train_model(subset, seed=seed)
        returns = evaluate_config(
            model, planner="random_shooting", horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, num_episodes=num_episodes, seed=seed,
        )
        results[size] = summarize(returns)
        results[size]["val_mse"] = history["best_val_loss"]
    return results


def collect_test_trajectories(
    model: DynamicsModel, num_mpc: int, num_random: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect held-out trajectories (states, actions) for prediction-error eval."""
    env = gym.make("CartPole-v1")
    trajectories = []

    for i in range(num_mpc):
        rng = np.random.default_rng(777_000 + seed + i)
        _, _, (states, actions, next_states) = run_episode(
            env, model, planner="random_shooting", horizon=DEFAULT_HORIZON,
            num_sequences=DEFAULT_NUM_SEQUENCES, rng=rng, reset_seed=777_000 + seed + i,
        )
        full_states = np.concatenate([states, next_states[-1:]])
        trajectories.append((full_states, actions))

    rng = np.random.default_rng(888_000 + seed)
    for i in range(num_random):
        state, _ = env.reset(seed=888_000 + seed + i)
        states, actions = [state], []
        done = False
        while not done:
            action = int(rng.integers(2))
            state, _, terminated, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            done = terminated or truncated
        trajectories.append((np.array(states), np.array(actions)))

    env.close()
    return trajectories


def multistep_prediction_error(
    models: dict[str, DynamicsModel],
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    max_horizon: int = 30,
) -> dict:
    """Mean open-loop L2 prediction error at each step k = 1..max_horizon.

    For every trajectory and every start index from which a full
    max_horizon-step rollout fits, the model is unrolled on the recorded
    actions and compared to the states the environment actually visited.
    """
    errors = {name: np.zeros(max_horizon) for name in models}
    counts = np.zeros(max_horizon)

    for states, actions in trajectories:
        num_steps = len(actions)
        horizon = min(max_horizon, num_steps)
        starts = np.arange(num_steps - horizon + 1)
        if len(starts) == 0:
            continue
        for name, model in models.items():
            current = states[starts]
            for k in range(1, horizon + 1):
                current = model.predict(current, actions[starts + k - 1])
                diff = np.linalg.norm(current - states[starts + k], axis=1)
                errors[name][k - 1] += diff.sum()
        counts[:horizon] += len(starts)

    valid = counts > 0
    return {
        "steps": list(range(1, int(valid.sum()) + 1)),
        **{
            name: (err[valid] / counts[valid]).tolist()
            for name, err in errors.items()
        },
    }
