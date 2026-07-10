"""Figure generation for the evaluation and ablation results.

All figures are written as PNG files into the given output directory
and are the ones included in the report. Colors are fixed per entity
(planner or model variant) so the same thing has the same color in
every figure.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "random_shooting": "#2a78d6",
    "cem": "#eb6834",
    "oracle": "#4a3aa7",
    "delta": "#1baf7a",
    "absolute": "#e34948",
    "neutral": "#6b6b66",
}
LABELS = {"random_shooting": "Random shooting", "cem": "CEM"}
MAX_RETURN = 500.0

plt.rcParams.update({
    "figure.dpi": 200,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def _new_axis(figsize=(4.0, 2.7)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name))
    plt.close(fig)


def plot_learning_curves(curves: dict, random_policy_mean: float, outdir: str) -> None:
    """curves[planner] = list over seeds of (env_steps, mean_returns)."""
    fig, ax = _new_axis()
    for planner, seed_curves in curves.items():
        steps = np.mean([c[0] for c in seed_curves], axis=0)
        returns = np.array([c[1] for c in seed_curves])
        mean, std = returns.mean(axis=0), returns.std(axis=0)
        ax.plot(steps, mean, marker="o", markersize=3.5, linewidth=2,
                color=COLORS[planner], label=LABELS[planner])
        ax.fill_between(steps, mean - std, mean + std, color=COLORS[planner], alpha=0.15)
    ax.axhline(MAX_RETURN, color=COLORS["neutral"], linewidth=1, linestyle="--")
    ax.axhline(random_policy_mean, color=COLORS["neutral"], linewidth=1, linestyle=":")
    ax.text(0.99, MAX_RETURN - 12, "max return", ha="right", va="top",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=COLORS["neutral"])
    ax.text(0.99, random_policy_mean + 8, "random policy", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=COLORS["neutral"])
    ax.set_xlabel("Real environment steps in training data")
    ax.set_ylabel("Mean episode return")
    ax.set_ylim(0, 540)
    ax.legend(loc="lower right")
    _save(fig, outdir, "learning_curve.png")


def _plot_sweep(ax, results: dict, x_values: list) -> None:
    for planner in results:
        means = [results[planner][x]["mean"] for x in x_values]
        stds = [results[planner][x]["std"] for x in x_values]
        ax.errorbar(x_values, means, yerr=stds, marker="o", markersize=3.5,
                    linewidth=2, capsize=3, color=COLORS[planner], label=LABELS[planner])
    ax.set_ylabel("Mean episode return")
    ax.set_ylim(0, 540)
    ax.axhline(MAX_RETURN, color=COLORS["neutral"], linewidth=1, linestyle="--")


def plot_horizon_ablation(results: dict, horizons: list[int], outdir: str) -> None:
    fig, ax = _new_axis()
    _plot_sweep(ax, results, horizons)
    ax.set_xticks(horizons)
    ax.set_xlabel("Planning horizon H")
    ax.legend(loc="lower right")
    _save(fig, outdir, "ablation_horizon.png")


def plot_sequences_ablation(results: dict, counts: list[int], outdir: str) -> None:
    fig, ax = _new_axis()
    _plot_sweep(ax, results, counts)
    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("Number of sampled action sequences N")
    ax.legend(loc="lower right")
    _save(fig, outdir, "ablation_sequences.png")


def plot_baselines(results: dict, outdir: str) -> None:
    order = ["random_policy", "learned_random_shooting", "oracle_random_shooting",
             "learned_cem", "oracle_cem"]
    labels = ["Random\npolicy", "RS\nlearned", "RS\ntrue dyn.", "CEM\nlearned", "CEM\ntrue dyn."]
    colors = [COLORS["neutral"], COLORS["random_shooting"], COLORS["oracle"],
              COLORS["cem"], COLORS["oracle"]]
    means = [results[k]["mean"] for k in order]
    stds = [results[k]["std"] for k in order]

    fig, ax = _new_axis()
    bars = ax.bar(labels, means, yerr=stds, capsize=3, color=colors, width=0.62)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                f"{mean:.0f}", ha="center", fontsize=8)
    ax.axhline(MAX_RETURN, color=COLORS["neutral"], linewidth=1, linestyle="--")
    ax.set_ylabel("Mean episode return")
    ax.set_ylim(0, 560)
    ax.grid(axis="x", visible=False)
    _save(fig, outdir, "baselines.png")


def plot_delta_vs_absolute(results: dict, outdir: str) -> None:
    fig, ax = _new_axis(figsize=(3.2, 2.7))
    names = ["delta", "absolute"]
    labels = ["Predict $\\Delta s$", "Predict $s_{t+1}$"]
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]
    bars = ax.bar(labels, means, yerr=stds, capsize=3,
                  color=[COLORS[n] for n in names], width=0.55)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 12,
                f"{mean:.0f}", ha="center", fontsize=8)
    ax.axhline(MAX_RETURN, color=COLORS["neutral"], linewidth=1, linestyle="--")
    ax.set_ylabel("Mean episode return")
    ax.set_ylim(0, 560)
    ax.grid(axis="x", visible=False)
    _save(fig, outdir, "delta_vs_absolute.png")


def plot_multistep_error(errors: dict, outdir: str) -> None:
    fig, ax = _new_axis(figsize=(3.6, 2.7))
    labels = {"delta": "Predict $\\Delta s$", "absolute": "Predict $s_{t+1}$"}
    for name in ("delta", "absolute"):
        ax.plot(errors["steps"], errors[name], linewidth=2,
                color=COLORS[name], label=labels[name])
    ax.set_yscale("log")
    ax.set_xlabel("Prediction steps k")
    ax.set_ylabel("Open-loop $L_2$ error")
    ax.legend(loc="lower right")
    _save(fig, outdir, "multistep_error.png")


def plot_capacity_and_data(size_results: dict, data_results: dict, outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)

    for ax, results, xlabel in (
        (axes[0], size_results, "Hidden layer size"),
        (axes[1], data_results, "Training transitions"),
    ):
        x_values = sorted(results)
        means = [results[x]["mean"] for x in x_values]
        stds = [results[x]["std"] for x in x_values]
        ax.errorbar(x_values, means, yerr=stds, marker="o", markersize=3.5,
                    linewidth=2, capsize=3, color=COLORS["random_shooting"])
        ax.set_xscale("log")
        ax.set_xticks(x_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.axhline(MAX_RETURN, color=COLORS["neutral"], linewidth=1, linestyle="--")
        ax.set_ylim(0, 540)
    axes[0].set_ylabel("Mean episode return")
    _save(fig, outdir, "capacity_and_data.png")


def plot_training_curve(history: dict, outdir: str) -> None:
    fig, ax = _new_axis(figsize=(3.6, 2.7))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], linewidth=2,
            color=COLORS["random_shooting"], label="Train")
    ax.plot(epochs, history["val_loss"], linewidth=2,
            color=COLORS["cem"], label="Validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.legend(loc="upper right")
    _save(fig, outdir, "training_curve.png")
