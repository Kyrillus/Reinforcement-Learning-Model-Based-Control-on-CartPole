"""Run the full pipeline: learning curves, all ablations, and figures.

Usage:
    python main.py            # full experiments (takes a while on CPU)
    python main.py --quick    # small smoke-test run of the whole pipeline

Writes figures to figures/ and all numerical results to results/results.json.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    parser.add_argument("--figdir", default="figures")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL

    start = time.time()
    results: dict = {"config": cfg}

    print("[1/4] Learning curves (train model, run MPC, aggregate data) ...")
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
    results["learning_curves"] = curves

    print("[2/4] Building the fixed ablation dataset and default model ...")
    dataset = build_ablation_dataset(target_size=cfg["dataset_size"])
    model, history = train_model(dataset)
    results["default_model_val_mse"] = history["best_val_loss"]
    plots.plot_training_curve(history, args.figdir)

    print("[3/4] Baselines and planner ablations ...")
    results["baselines"] = baselines(model, cfg["eval_episodes"], seed=0)
    print(f"    baselines: " + ", ".join(
        f"{k}={v['mean']:.0f}" for k, v in results["baselines"].items()))
    results["horizon"] = ablation_horizon(model, cfg["horizons"], cfg["eval_episodes"], seed=0)
    results["num_sequences"] = ablation_num_sequences(
        model, cfg["sequence_counts"], cfg["eval_episodes"], seed=0)

    print("[4/4] Model ablations (delta vs. absolute, size, data amount) ...")
    results["delta_vs_absolute"], variant_models = ablation_delta_vs_absolute(
        dataset, cfg["eval_episodes"], seed=0)
    trajectories = collect_test_trajectories(model, num_mpc=3, num_random=10, seed=0)
    results["multistep_error"] = multistep_prediction_error(variant_models, trajectories)
    results["model_size"] = ablation_model_size(
        dataset, cfg["hidden_sizes"], cfg["eval_episodes"], seed=0)
    results["data_amount"] = ablation_data_amount(
        dataset, cfg["data_sizes"], cfg["eval_episodes"], seed=0)

    random_policy_mean = results["baselines"]["random_policy"]["mean"]
    plots.plot_learning_curves(curves, random_policy_mean, args.figdir)
    plots.plot_baselines(results["baselines"], args.figdir)
    plots.plot_horizon_ablation(results["horizon"], cfg["horizons"], args.figdir)
    plots.plot_sequences_ablation(results["num_sequences"], cfg["sequence_counts"], args.figdir)
    plots.plot_delta_vs_absolute(results["delta_vs_absolute"], args.figdir)
    plots.plot_multistep_error(results["multistep_error"], args.figdir)
    plots.plot_capacity_and_data(results["model_size"], results["data_amount"], args.figdir)

    os.makedirs(args.outdir, exist_ok=True)
    results["runtime_seconds"] = round(time.time() - start, 1)
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done in {results['runtime_seconds']:.0f}s. "
          f"Figures in {args.figdir}/, results in {args.outdir}/results.json")


if __name__ == "__main__":
    main()
