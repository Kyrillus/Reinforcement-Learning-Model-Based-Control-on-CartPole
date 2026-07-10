"""Run the full pipeline: learning curves (training), all ablations, and figures.

Usage:
    python main.py                    # everything: training/learning curves + ablations
    python main.py --learning-curve   # only the training experiment (learning curves)
    python main.py --ablations        # only the ablations
    python main.py --quick            # small smoke-test run (combinable with the above)

Writes figures to figures/ and numerical results to results/results.json.
When a stage is run on its own, its results are merged into an existing
results.json instead of replacing it.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ablation import plots
from ablation.ablation import (
    ablation_data_amount,
    ablation_delta_vs_absolute,
    ablation_horizon,
    ablation_model_size,
    ablation_num_sequences,
    baselines,
    build_ablation_dataset,
    collect_test_trajectories,
    multistep_prediction_error,
    random_policy_returns,
    train_model,
)
from evaluation.evaluate import run_learning_curve

FULL = {
    "lc_seeds": [0, 1, 2],
    "lc_iterations": 5,
    "lc_init_random_steps": 1000,
    "lc_episodes_per_iter": 3,
    "eval_episodes": 15,
    "dataset_size": 8000,
    "horizons": [10, 20, 30],
    "sequence_counts": [50, 200, 500],
    "hidden_sizes": [16, 64, 256],
    "data_sizes": [500, 2000, 8000],
}

QUICK = {
    "lc_seeds": [0],
    "lc_iterations": 2,
    "lc_init_random_steps": 300,
    "lc_episodes_per_iter": 1,
    "eval_episodes": 2,
    "dataset_size": 800,
    "horizons": [10, 20],
    "sequence_counts": [50, 200],
    "hidden_sizes": [16, 64],
    "data_sizes": [300, 800],
}


def run_learning_curves(cfg: dict, figdir: str) -> dict:
    """The training experiment: train the model, run MPC, aggregate data."""
    print("Learning curves (train model, run MPC, aggregate data) ...")
    curves: dict = {}
    for planner in ("random_shooting", "cem"):
        curves[planner] = []
        for seed in cfg["lc_seeds"]:
            env_steps, mean_returns, _ = run_learning_curve(
                planner, seed=seed,
                iterations=cfg["lc_iterations"],
                init_random_steps=cfg["lc_init_random_steps"],
                episodes_per_iter=cfg["lc_episodes_per_iter"],
            )
            curves[planner].append((env_steps, mean_returns))
            print(f"    {planner} seed {seed}: returns per iteration {mean_returns}")

    random_policy_mean = float(np.mean(random_policy_returns(cfg["eval_episodes"], seed=0)))
    plots.plot_learning_curves(curves, random_policy_mean, figdir)
    return {"learning_curves": curves}


def run_ablations(cfg: dict, figdir: str) -> dict:
    """All ablations and baselines on a fixed dataset."""
    results: dict = {}

    print("Building the fixed ablation dataset and default model ...")
    dataset = build_ablation_dataset(target_size=cfg["dataset_size"])
    model, history = train_model(dataset)
    results["default_model_val_mse"] = history["best_val_loss"]
    plots.plot_training_curve(history, figdir)

    print("Baselines and planner ablations ...")
    results["baselines"] = baselines(model, cfg["eval_episodes"], seed=0)
    print(f"    baselines: " + ", ".join(
        f"{k}={v['mean']:.0f}" for k, v in results["baselines"].items()))
    results["horizon"] = ablation_horizon(model, cfg["horizons"], cfg["eval_episodes"], seed=0)
    results["num_sequences"] = ablation_num_sequences(
        model, cfg["sequence_counts"], cfg["eval_episodes"], seed=0)

    print("Model ablations (delta vs. absolute, size, data amount) ...")
    results["delta_vs_absolute"], variant_models = ablation_delta_vs_absolute(
        dataset, cfg["eval_episodes"], seed=0)
    trajectories = collect_test_trajectories(model, num_mpc=3, num_random=10, seed=0)
    results["multistep_error"] = multistep_prediction_error(variant_models, trajectories)
    results["model_size"] = ablation_model_size(
        dataset, cfg["hidden_sizes"], cfg["eval_episodes"], seed=0)
    results["data_amount"] = ablation_data_amount(
        dataset, cfg["data_sizes"], cfg["eval_episodes"], seed=0)

    plots.plot_baselines(results["baselines"], figdir)
    plots.plot_horizon_ablation(results["horizon"], cfg["horizons"], figdir)
    plots.plot_sequences_ablation(results["num_sequences"], cfg["sequence_counts"], figdir)
    plots.plot_delta_vs_absolute(results["delta_vs_absolute"], figdir)
    plots.plot_multistep_error(results["multistep_error"], figdir)
    plots.plot_capacity_and_data(results["model_size"], results["data_amount"], figdir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--learning-curve", action="store_true",
                        help="run only the training experiment (learning curves)")
    parser.add_argument("--ablations", action="store_true",
                        help="run only the ablations")
    parser.add_argument("--figdir", default="figures")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    run_all = not (args.learning_curve or args.ablations)

    start = time.time()
    results: dict = {"config": cfg}

    if run_all or args.learning_curve:
        results.update(run_learning_curves(cfg, args.figdir))
    if run_all or args.ablations:
        results.update(run_ablations(cfg, args.figdir))

    # Merge into an existing results.json so single-stage runs keep the
    # results of the other stage.
    os.makedirs(args.outdir, exist_ok=True)
    results_path = os.path.join(args.outdir, "results.json")
    merged: dict = {}
    if not run_all and os.path.exists(results_path):
        with open(results_path) as f:
            merged = json.load(f)
    merged.update(results)
    merged["runtime_seconds"] = round(time.time() - start, 1)
    with open(results_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Done in {merged['runtime_seconds']:.0f}s. "
          f"Figures in {args.figdir}/, results in {results_path}")


if __name__ == "__main__":
    main()
