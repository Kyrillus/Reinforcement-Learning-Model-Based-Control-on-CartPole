# Model-Based Control on CartPole

Model-based RL on CartPole-v1: a learned MLP dynamics model with MPC planning
(random shooting and CEM), following Nagabandi et al. (arXiv:1708.02596).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py                    # everything: training (learning curves) + all ablations
python main.py --learning-curve   # only the training experiment (learning curves)
python main.py --ablations        # only the ablations
```

Figures are written to `figures/`, numerical results to `results/results.json`
(single-stage runs merge into an existing results.json instead of replacing it).
The ablations cover planning horizon, number of sampled action sequences,
random shooting vs. CEM, delta vs. absolute state prediction, and
dynamics-model size / training data amount.

## Tests

```bash
python -m pytest tests/
```

## Report

The project report is in `report/` (`report.pdf`, built from `report.tex`
with the NeurIPS 2024 style). To rebuild it after rerunning the experiments:

```bash
cp figures/*.png report/figures/ && cd report && tectonic report.tex
```
