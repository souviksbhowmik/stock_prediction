"""Deep Q-Network (DQN) agent for stock trading.

Design notes
------------
State space
    Continuous feature vector from the **last timestep** of the scaled
    sequence window (``X_seq[:, -1, :]``).  No discretisation is required —
    the neural Q-network generalises to unseen feature combinations through
    non-linear function approximation, unlike the tabular Q-table.

Action space
    0 = SELL  (close / stay flat)
    1 = HOLD  (keep current position)
    2 = BUY   (open / stay long)

Network architecture
    Input(n_features) → [Linear → ReLU → Dropout] × n_layers → Linear(3)
    Huber (smooth-L1) loss for robust training against occasional outlier
    rewards.

DQN improvements over tabular Q-learning
    • Neural Q-function — no state binning; smooth generalisation across the
      continuous feature space.
    • Experience replay — random mini-batch sampling from a circular buffer
      breaks temporal correlations and recycles past experience.
    • Target network — a periodically-frozen copy of the online Q-network
      supplies stable Bellman targets, reducing the divergence risk from the
      moving-target problem.
    • Gradient clipping — prevents exploding gradients from large reward
      spikes.

Reward function  (Moody & Saffell 2001; Spooner et al. 2018)
    Same position-based P&L reward as the tabular Q-learning model:

        reward(t) = new_position(t) × r(t) − |Δposition| × tc

    where
        r(t)            = 1-step price return at step t
        new_position(t) = 1 after BUY, 0 after SELL, unchanged after HOLD
        Δposition       = |new_position − old_position|
        tc              = transaction_cost (default 0.1 %)

Input at training / inference
    Uses the **last timestep** of the seq-scaler-normalised sequence window,
    i.e. ``X_seq[:, -1, :]`` — exactly what the Ensemble passes as X_seq.

Probability output
    Softmax over Q-values with a tunable temperature.
    Returns uniform prior when the model has not been trained yet.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.dqn")


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Neural network ────────────────────────────────────────────────────────────

class _QNetwork(nn.Module):
    """MLP that maps a feature vector to Q-values for 3 actions (SELL/HOLD/BUY)."""

    def __init__(self, input_size: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    """Fixed-capacity circular buffer for experience replay."""

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, float(reward), next_state, float(done)))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── Main predictor ────────────────────────────────────────────────────────────

class DQNPredictor:
    """Deep Q-Network agent for stock Buy / Hold / Sell decisions.

    Follows the same ``train`` / ``predict_proba`` / ``predict`` /
    ``save`` / ``load`` API as all other predictors in this project.

    Parameters
    ----------
    input_size:
        Number of features in the input state vector.
    hidden_sizes:
        Sizes of the hidden MLP layers (e.g. [256, 128]).
    dropout:
        Dropout probability applied after each hidden layer.
    learning_rate:
        Adam optimiser step size.
    batch_size:
        Number of transitions sampled from the replay buffer per update.
    buffer_size:
        Maximum capacity of the replay buffer.
    target_update_freq:
        Number of gradient steps between target-network syncs.
    discount_factor:
        Future reward discount γ.
    epsilon_start / epsilon_end / epsilon_decay:
        ε-greedy schedule.  ε_episode = max(ε_end, ε_start × ε_decay^episode).
    n_episodes:
        How many passes over the training time-series.
    transaction_cost:
        Fraction of trade value charged per open/close (e.g. 0.001 = 0.1 %).
    temperature:
        Softmax temperature for converting Q-values to probabilities.
    horizon:
        Forecast horizon (used only to look up signal thresholds for logging).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        buffer_size: int | None = None,
        target_update_freq: int | None = None,
        discount_factor: float | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay: float | None = None,
        n_episodes: int | None = None,
        transaction_cost: float | None = None,
        temperature: float | None = None,
        horizon: int | None = None,
    ):
        self.input_size = input_size

        def _cfg(key: str, default):
            return get_setting("models", "dqn", key, default=default)

        self.hidden_sizes      = list(hidden_sizes) if hidden_sizes is not None else list(_cfg("hidden_sizes", [256, 128]))
        self.dropout           = dropout            if dropout            is not None else float(_cfg("dropout", 0.2))
        self.lr                = learning_rate      if learning_rate      is not None else float(_cfg("learning_rate", 0.001))
        self.batch_size        = batch_size         if batch_size         is not None else int(_cfg("batch_size", 64))
        self.buffer_size       = buffer_size        if buffer_size        is not None else int(_cfg("buffer_size", 10000))
        self.target_update_freq = target_update_freq if target_update_freq is not None else int(_cfg("target_update_freq", 100))
        self.gamma             = discount_factor    if discount_factor    is not None else float(_cfg("discount_factor", 0.99))
        self.epsilon_start     = epsilon_start      if epsilon_start      is not None else float(_cfg("epsilon_start", 1.0))
        self.epsilon_end       = epsilon_end        if epsilon_end        is not None else float(_cfg("epsilon_end", 0.01))
        self.epsilon_decay     = epsilon_decay      if epsilon_decay      is not None else float(_cfg("epsilon_decay", 0.995))
        self.n_episodes        = n_episodes         if n_episodes         is not None else int(_cfg("n_episodes", 10))
        self.transaction_cost  = transaction_cost   if transaction_cost   is not None else float(_cfg("transaction_cost", 0.001))
        self.temperature       = temperature        if temperature        is not None else float(_cfg("temperature", 0.5))
        self.horizon           = horizon            if horizon            is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )
        # Minimum replay buffer size before training begins
        raw_min = int(_cfg("min_buffer_size", self.batch_size))
        self.min_buffer_size   = max(raw_min, self.batch_size)

        self._device   = _get_device()
        self._trained  = False

        # Networks built lazily in train() to allow load() to rebuild correctly
        self._online_net: _QNetwork | None = None
        self._target_net: _QNetwork | None = None
        self._optimizer: optim.Optimizer | None = None

    # ── Network management ────────────────────────────────────────────────────

    def _build_networks(self) -> None:
        self._online_net = _QNetwork(
            self.input_size, self.hidden_sizes, self.dropout
        ).to(self._device)
        self._target_net = _QNetwork(
            self.input_size, self.hidden_sizes, self.dropout
        ).to(self._device)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(self._online_net.parameters(), lr=self.lr)

    def _sync_target(self) -> None:
        """Copy online network weights to the target network."""
        self._target_net.load_state_dict(self._online_net.state_dict())

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_tab: np.ndarray,           # (N, n_features)  last-timestep features
        returns_1d: np.ndarray,      # (N,)  1-step price return per step
        labels: np.ndarray | None = None,   # (N,)  true labels (0/1/2) for logging
        feature_names: list[str] | None = None,  # accepted for API parity; unused
    ) -> dict:
        """Train the DQN agent over chronological time-series data.

        Parameters
        ----------
        X_tab:
            Scaled feature matrix, shape (N, n_features).  This should be the
            *last* timestep of the sequence window — i.e. the current-bar
            observation.
        returns_1d:
            1-step price return r(t) = close[t+1]/close[t] − 1, shape (N,).
        labels:
            Optional true signal labels (0=SELL,1=HOLD,2=BUY) for episode-end
            balanced-accuracy logging.
        feature_names:
            Accepted for API parity with other predictors; not used by DQN.
        """
        self._build_networks()
        buffer = _ReplayBuffer(self.buffer_size)

        N = len(X_tab)
        history: dict = {"episode_rewards": [], "episode_balanced_acc": []}
        total_steps = 0

        self._online_net.train()

        for episode in range(self.n_episodes):
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start * (self.epsilon_decay ** episode),
            )
            position = 0
            total_reward = 0.0
            pred_labels: list[int] = []

            for t in range(N - 1):
                state = X_tab[t].astype(np.float32)

                # ε-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(
                            state, dtype=torch.float32, device=self._device
                        ).unsqueeze(0)
                        q_vals = self._online_net(s_t).squeeze(0).cpu().numpy()
                    action = int(np.argmax(q_vals))

                # Resolve new position from action
                if action == 2:      # BUY
                    new_position = 1
                elif action == 0:    # SELL
                    new_position = 0
                else:                # HOLD
                    new_position = position

                # Position-based reward with transaction cost
                r   = float(returns_1d[t])
                tc  = self.transaction_cost
                reward = new_position * r - abs(new_position - position) * tc

                next_state = X_tab[t + 1].astype(np.float32)
                done = (t == N - 2)  # True at the last valid transition

                buffer.push(state, action, reward, next_state, done)

                position     = new_position
                total_reward += reward
                pred_labels.append(action)
                total_steps  += 1

                # Gradient update once buffer is sufficiently filled
                if len(buffer) >= self.min_buffer_size:
                    self._update(buffer)

                # Sync target network every `target_update_freq` gradient steps
                if total_steps % self.target_update_freq == 0:
                    self._sync_target()

            avg_reward = total_reward / max(N - 1, 1)
            history["episode_rewards"].append(avg_reward)

            log_interval = max(1, self.n_episodes // 5)
            if labels is not None and len(labels) > 0:
                n = min(len(pred_labels), len(labels) - 1)
                ba = balanced_accuracy_score(labels[:n], pred_labels[:n])
                history["episode_balanced_acc"].append(ba)
                if (episode + 1) % log_interval == 0:
                    logger.info(
                        f"DQN episode {episode+1}/{self.n_episodes} — "
                        f"ε={epsilon:.3f}, avg_reward={avg_reward:.5f}, "
                        f"balanced_acc={ba:.4f}, buffer={len(buffer)}"
                    )
            else:
                if (episode + 1) % log_interval == 0:
                    logger.info(
                        f"DQN episode {episode+1}/{self.n_episodes} — "
                        f"ε={epsilon:.3f}, avg_reward={avg_reward:.5f}, "
                        f"buffer={len(buffer)}"
                    )

        self._trained = True
        self._online_net.eval()
        logger.info(
            f"DQN training complete — "
            f"architecture={self.hidden_sizes}, buffer={len(buffer)}"
        )
        return history

    def _update(self, buffer: _ReplayBuffer) -> None:
        """One Bellman gradient step on a random mini-batch."""
        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        s  = torch.tensor(states,      dtype=torch.float32, device=self._device)
        a  = torch.tensor(actions,     dtype=torch.int64,   device=self._device)
        r  = torch.tensor(rewards,     dtype=torch.float32, device=self._device)
        ns = torch.tensor(next_states, dtype=torch.float32, device=self._device)
        d  = torch.tensor(dones,       dtype=torch.float32, device=self._device)

        # Q(s, a) for the actions actually taken
        q_pred = self._online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ · max_a' Q_target(s', a') · (1 − done)
        with torch.no_grad():
            q_next   = self._target_net(ns).max(dim=1).values
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = nn.functional.huber_loss(q_pred, q_target)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online_net.parameters(), max_norm=10.0)
        self._optimizer.step()

    # ── Evaluation ────────────────────────────────────────────────────────────

    def compute_balanced_accuracy(
        self, X_tab: np.ndarray, labels: np.ndarray
    ) -> float:
        """Balanced accuracy of greedy-policy predictions vs true labels."""
        preds = self.predict(X_tab)
        n = min(len(preds), len(labels))
        return float(balanced_accuracy_score(labels[:n], preds[:n]))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert Q-values to class probabilities via temperature softmax.

        Accepts either:
        - 2-D ``(N, n_features)``    — tabular last-step features
        - 3-D ``(N, T, n_features)`` — sequence; uses the last timestep

        Returns ``(N, 3)`` float32 array; column order: SELL / HOLD / BUY.
        """
        if not self._trained or self._online_net is None:
            raise RuntimeError("DQN model has not been trained yet.")

        if X.ndim == 3:
            X = X[:, -1, :]   # take last timestep

        self._online_net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=self._device)
            q_vals = self._online_net(t).cpu().numpy()  # (N, 3)

        probs = np.empty((len(X), 3), dtype=np.float32)
        for i, q in enumerate(q_vals):
            probs[i] = softmax(q / max(self.temperature, 1e-8)).astype(np.float32)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels 0=SELL / 1=HOLD / 2=BUY."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict":  self._online_net.state_dict(),
                "input_size":         self.input_size,
                "hidden_sizes":       self.hidden_sizes,
                "dropout":            self.dropout,
                "learning_rate":      self.lr,
                "batch_size":         self.batch_size,
                "buffer_size":        self.buffer_size,
                "target_update_freq": self.target_update_freq,
                "discount_factor":    self.gamma,
                "epsilon_start":      self.epsilon_start,
                "epsilon_end":        self.epsilon_end,
                "epsilon_decay":      self.epsilon_decay,
                "n_episodes":         self.n_episodes,
                "transaction_cost":   self.transaction_cost,
                "temperature":        self.temperature,
                "horizon":            self.horizon,
                "min_buffer_size":    self.min_buffer_size,
                "trained":            self._trained,
            },
            path,
        )
        logger.info(f"Saved DQN model ({self.hidden_sizes}) to {path}")

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self.input_size         = data.get("input_size",         self.input_size)
        self.hidden_sizes       = data.get("hidden_sizes",       self.hidden_sizes)
        self.dropout            = data.get("dropout",            self.dropout)
        self.lr                 = data.get("learning_rate",      self.lr)
        self.batch_size         = data.get("batch_size",         self.batch_size)
        self.buffer_size        = data.get("buffer_size",        self.buffer_size)
        self.target_update_freq = data.get("target_update_freq", self.target_update_freq)
        self.gamma              = data.get("discount_factor",    self.gamma)
        self.epsilon_start      = data.get("epsilon_start",      self.epsilon_start)
        self.epsilon_end        = data.get("epsilon_end",        self.epsilon_end)
        self.epsilon_decay      = data.get("epsilon_decay",      self.epsilon_decay)
        self.n_episodes         = data.get("n_episodes",         self.n_episodes)
        self.transaction_cost   = data.get("transaction_cost",   self.transaction_cost)
        self.temperature        = data.get("temperature",        self.temperature)
        self.horizon            = data.get("horizon",            self.horizon)
        self.min_buffer_size    = data.get("min_buffer_size",    self.min_buffer_size)
        self._trained           = data.get("trained",            False)

        self._build_networks()
        self._online_net.load_state_dict(data["online_state_dict"])
        self._online_net.eval()
        logger.info(f"Loaded DQN model ({self.hidden_sizes}) from {path}")
