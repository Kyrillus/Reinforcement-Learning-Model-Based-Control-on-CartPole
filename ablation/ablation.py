"""Planner ablations: horizon H and number of sampled action sequences.

Optional deliverable from the project sheet (section 2d). Sweeps over
a grid of (horizon, num_sequences) values and plots average episode
return for each configuration.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from dynamics.base import DynamicsModel
from dynamics.stub_model import StubDynamicsModel
from evaluation.evaluate import evaluate


def run_ablation(
    model: DynamicsModel,
    horizons: list[int],
    num_sequences_list: list[int],
    planner: str = "cem",
    num_episodes: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Sweep over (horizon, num_sequences) and record mean episode return.

    Returns:
        Array of shape (len(horizons), len(num_sequences_list)) with mean returns.
    """
    results = np.zeros((len(horizons), len(num_sequences_list)))
    for i, horizon in enumerate(horizons):
        for j, num_sequences in enumerate(num_sequences_list):
            _, returns = evaluate(
                model,
                num_episodes=num_episodes,
                planner=planner,
                horizon=horizon,
                num_sequences=num_sequences,
                seed=seed,
            )
            results[i, j] = float(np.mean(returns))
    return results


def plot_ablation(
    results: np.ndarray,
    horizons: list[int],
    num_sequences_list: list[int],
    save_path: str = "ablation/ablation_heatmap.png",
) -> None:
    """Plot a heatmap of mean return over the (horizon, num_sequences) grid."""
    plt.figure(figsize=(7, 5))
    plt.imshow(results, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Mean episode return")
    plt.xticks(range(len(num_sequences_list)), num_sequences_list)
    plt.yticks(range(len(horizons)), horizons)
    plt.xlabel("Number of sampled sequences")
    plt.ylabel("Horizon H")
    plt.title("Planner ablation: H vs. num. sequences")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    stub_model = StubDynamicsModel()
    horizons = [10, 20, 30]
    num_sequences_list = [50, 200]
    results = run_ablation(stub_model, horizons, num_sequences_list, num_episodes=3)
    plot_ablation(results, horizons, num_sequences_list)
    print(results)
