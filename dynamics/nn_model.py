"""Learned neural dynamics model for CartPole.

An MLP takes the current state and a one-hot encoded action and predicts
either the state change (delta) or the absolute next state. Inputs and
targets are standardized with statistics computed from the training data,
as in Nagabandi et al. (2018). Trained with an MSE loss, a train/val
split, and early stopping on the validation loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dynamics.base import DynamicsModel


class NeuralDynamicsModel(DynamicsModel):
    """MLP dynamics model f(s_t, a_t) -> s_{t+1} (via delta or absolute)."""

    def __init__(
        self,
        state_dim: int = 4,
        num_actions: int = 2,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        predict_delta: bool = True,
        seed: int = 0,
    ) -> None:
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.predict_delta = predict_delta

        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        in_dim = state_dim + num_actions
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

        # Normalization statistics, overwritten in fit().
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_std = np.ones(state_dim, dtype=np.float32)
        self.target_mean = np.zeros(state_dim, dtype=np.float32)
        self.target_std = np.ones(state_dim, dtype=np.float32)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def _make_inputs(self, states: np.ndarray, actions: np.ndarray) -> torch.Tensor:
        normed = (states - self.state_mean) / self.state_std
        one_hot = np.eye(self.num_actions, dtype=np.float32)[actions.astype(int)]
        return torch.from_numpy(
            np.concatenate([normed.astype(np.float32), one_hot], axis=1)
        )

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_frac: float = 0.1,
        patience: int = 10,
        seed: int = 0,
    ) -> dict:
        """Train on transitions with MSE, early stopping on a validation split.

        Returns:
            Dict with per-epoch "train_loss" / "val_loss" lists and the
            "best_val_loss" (normalized MSE) of the restored parameters.
        """
        targets = next_states - states if self.predict_delta else next_states

        self.state_mean = states.mean(axis=0).astype(np.float32)
        self.state_std = (states.std(axis=0) + 1e-6).astype(np.float32)
        self.target_mean = targets.mean(axis=0).astype(np.float32)
        self.target_std = (targets.std(axis=0) + 1e-6).astype(np.float32)

        inputs = self._make_inputs(states, actions)
        targets_t = torch.from_numpy(
            ((targets - self.target_mean) / self.target_std).astype(np.float32)
        )

        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(inputs))
        num_val = max(1, int(len(inputs) * val_frac))
        val_idx, train_idx = perm[:num_val], perm[num_val:]
        train_x, train_y = inputs[train_idx], targets_t[train_idx]
        val_x, val_y = inputs[val_idx], targets_t[val_idx]

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        history: dict = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        epochs_since_best = 0

        for _ in range(epochs):
            order = torch.from_numpy(rng.permutation(len(train_x)))
            epoch_losses = []
            for start in range(0, len(train_x), batch_size):
                idx = order[start : start + batch_size]
                pred = self.net(train_x[idx])
                loss = loss_fn(pred, train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            with torch.no_grad():
                val_loss = loss_fn(self.net(val_x), val_y).item()
            history["train_loss"].append(float(np.mean(epoch_losses)))
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    break

        self.net.load_state_dict(best_state)
        history["best_val_loss"] = best_val
        return history

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.net(self._make_inputs(states, actions)).numpy()
        out = out * self.target_std + self.target_mean
        return states + out if self.predict_delta else out
