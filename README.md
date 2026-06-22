# Project 23: Model-Based Control on CartPole

192.151 Introduction to Deep Learning — Person B's part: MPC Planner + Evaluation.

## Scope of this part (Person B)

- `mpc/random_shooting.py` — random shooting baseline planner.
- `mpc/cem.py` — Cross-Entropy Method (CEM) planner over discrete action sequences.
- `mpc/controller.py` — receding-horizon control loop (plan, execute first action, replan).
- `evaluation/evaluate.py` — evaluation harness: episode return vs. environment steps, learning-curve plot.
- `ablation/ablation.py` — ablations over horizon `H` and number of sampled sequences.
- `dynamics/base.py` — the `DynamicsModel` interface the planner expects, plus CartPole-v1's reward/termination
  logic (needed to score imagined rollouts).
- `dynamics/stub_model.py` — a temporary true-physics stand-in implementing `DynamicsModel`, used to develop and
  test the planner before the real learned model exists.

## Not in this part (Person A)

The actual learned neural dynamics model (data collection from the real environment + MLP training with a
train/val split) is Person A's deliverable. Once that model implements the `DynamicsModel` interface in
`dynamics/base.py`, swap it in for `StubDynamicsModel` in `evaluation/evaluate.py` and `ablation/ablation.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python -m evaluation.evaluate   # produces evaluation/learning_curve.png
python -m ablation.ablation     # produces ablation/ablation_heatmap.png
python -m pytest tests/         # sanity tests (planner runs end-to-end)
```

## Reference

Nagabandi et al., "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model Predictive
Control" (2018). https://arxiv.org/pdf/1708.02596
