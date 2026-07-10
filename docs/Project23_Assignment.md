# 192.151 Introduction to Deep Learning — 2026S

## Project 23: Reinforcement Learning — Model-Based Control on CartPole

## 1. Background and Motivation

Model-free RL methods (e.g., policy gradients) can learn strong policies, but often require many environment interactions. In contrast, model-based control learns a predictive model of environment dynamics from collected transitions, and then uses planning to select actions. This can be substantially more sample-efficient, since a learned model can be queried many times during planning without additional real environment steps. A practical and widely-used planning approach is **Model Predictive Control (MPC)**: at each time step, the agent plans a short action sequence using the learned model, executes only the first action, then replans from the new state.

## 2. Problem Description

The primary goal is to solve **CartPole-v1** by learning a dynamics model from data and controlling the system using MPC.

### a) Environment

- Use CartPole-v1 (discrete actions).

### b) Dynamics Model

Learn a predictive model $\hat{f}_\theta$ of the environment:

- Implement a neural dynamics model (recommended: MLP) predicting either next state $s_{t+1}$ or state delta $\Delta s_t$ given $(s_t, a_t)$.
- Train with a supervised loss (e.g., MSE), and use a standard train/val split for early stopping or hyperparameter selection.

### c) MPC Planner

Implement planning using the learned dynamics model:

- At each time step, plan an action sequence of horizon $H$ (e.g., $H \in [10, 30]$) that maximizes predicted cumulative reward.
- Use a simple optimizer such as random shooting or CEM (Cross-Entropy Method) over discrete action sequences.
- Execute the first action, observe the next state, and replan (receding-horizon control).

### d) Evaluation

- **Primary metric:** average episode return vs. environment steps (learning curve).
- *(Optional)* Planner ablation: horizon $H$ and/or number of sampled action sequences.

## 3. Expected Deliverables

Students working in groups (ideally 3 members) are expected to:

- produce a project report,
- deliver a group presentation,
- submit well-commented source code (data collection + model training + MPC planner + evaluation/plots).

## 4. Suggested Resources and References

[1] Nagabandi, A., Kahn, G., Fearing, R. S., & Levine, S. (2018). *Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model Predictive Control.* arXiv preprint arXiv:1708.02596. <https://arxiv.org/pdf/1708.02596>

[2] Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models.* In NeurIPS. <https://arxiv.org/pdf/1805.12114>

[3] OpenAI Gym / Gymnasium (CartPole environment). <https://www.gymlibrary.dev/>