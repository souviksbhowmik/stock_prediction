"""Deep Q-Network (DQN) agent for stock trading.

Design notes
------------
State space
    By default: full 60-step sequence window passed through a single-layer GRU
    encoder, which compresses it to a fixed-size context vector.  When
    ``encoder_hidden_size`` is not set, falls back to the **last timestep** of
    the scaled sequence window (``X_seq[:, -1, :]``) for backward compatibility.

    The GRU encoder eliminates *state aliasing*: two situations that share the
    same current-day features but differ in their recent history now produce
    distinct state vectors, allowing the Q-network to learn history-dependent
    trading strategies.

Action space
    0 = SELL  (close / stay flat)
    1 = HOLD  (keep current position)
    2 = BUY   (open / stay long)

Network architecture
    With GRU encoder (default when encoder_hidden_size is set):
        GRU(input_size → encoder_hidden_size) →
        [Linear → ReLU → Dropout] × n_layers → Linear(3)

    Without encoder (legacy / backward-compat):
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
    3-D ``(N, seq_len, n_features)`` — full sequence window fed through the
    GRU encoder.  When the encoder is disabled, 2-D ``(N, n_features)`` is
    also accepted (last-timestep mode).  Passing 3-D to a non-encoder model
    automatically strips to the last timestep for backward compatibility.

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


# ── Neural network components ─────────────────────────────────────────────────

class _GRUEncoder(nn.Module):
    """Single-layer GRU: (B, seq_len, input_size) → (B, hidden_size).

    Compresses the full sequence history into a fixed-size context vector.
    The final hidden state captures temporal patterns across all timesteps,
    giving the Q-network access to the complete 60-day price/indicator history.
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)                 # h: (1, B, hidden_size)
        return self.dropout(h.squeeze(0))  # → (B, hidden_size)


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


class _DQNNetwork(nn.Module):
    """Optional GRU encoder + MLP Q-head bundled as one nn.Module.

    Bundling encoder and Q-head means ``_sync_target()`` copies encoder
    weights automatically via ``load_state_dict()`` — no extra bookkeeping.
    When ``encoder`` is None the module behaves as a plain Q-network.
    """

    def __init__(self, encoder: _GRUEncoder | None, q_head: _QNetwork):
        super().__init__()
        self.q_head = q_head
        # Only register encoder as a submodule when it exists so that
        # state_dict() / load_state_dict() remain schema-consistent.
        if encoder is not None:
            self.encoder: _GRUEncoder | None = encoder
        else:
            self.encoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(x)    # (B, T, F) → (B, hidden_size)
        return self.q_head(x)      # (B, hidden_size or n_features) → (B, 3)


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
        Number of features per timestep in the input sequence.
    hidden_sizes:
        Sizes of the hidden MLP Q-head layers (e.g. [256, 128]).
    encoder_hidden_size:
        If set, a single-layer GRU encoder is added before the Q-head.
        The GRU compresses the full sequence to a vector of this size.
        When ``None`` (default) the model uses only the last timestep.
    dropout:
        Dropout probability applied after each hidden layer and the encoder.
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
    epsilon_start / epsilon_end:
        ε-greedy linear schedule.  ε decays linearly from ε_start to ε_end
        over the total number of environment steps (N × n_episodes).
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
        encoder_hidden_size: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        buffer_size: int | None = None,
        target_update_freq: int | None = None,
        discount_factor: float | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        n_episodes: int | None = None,
        transaction_cost: float | None = None,
        temperature: float | None = None,
        horizon: int | None = None,
    ):
        self.input_size = input_size

        def _cfg(key: str, default):
            return get_setting("models", "dqn", key, default=default)

        self.hidden_sizes        = list(hidden_sizes) if hidden_sizes is not None else list(_cfg("hidden_sizes", [256, 128]))
        self.encoder_hidden_size = encoder_hidden_size   # None → no encoder
        self._use_encoder        = encoder_hidden_size is not None
        self.dropout             = dropout           if dropout            is not None else float(_cfg("dropout", 0.2))
        self.lr                  = learning_rate     if learning_rate      is not None else float(_cfg("learning_rate", 0.001))
        self.batch_size          = batch_size        if batch_size         is not None else int(_cfg("batch_size", 64))
        self.buffer_size         = buffer_size       if buffer_size        is not None else int(_cfg("buffer_size", 10000))
        self.target_update_freq  = target_update_freq if target_update_freq is not None else int(_cfg("target_update_freq", 100))
        self.gamma               = discount_factor   if discount_factor    is not None else float(_cfg("discount_factor", 0.99))
        self.epsilon_start       = epsilon_start     if epsilon_start      is not None else float(_cfg("epsilon_start", 1.0))
        self.epsilon_end         = epsilon_end       if epsilon_end        is not None else float(_cfg("epsilon_end", 0.01))
        self.n_episodes          = n_episodes        if n_episodes         is not None else int(_cfg("n_episodes", 10))
        self.transaction_cost    = transaction_cost  if transaction_cost   is not None else float(_cfg("transaction_cost", 0.001))
        self.temperature         = temperature       if temperature        is not None else float(_cfg("temperature", 0.5))
        self.horizon             = horizon           if horizon            is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )
        raw_min = int(_cfg("min_buffer_size", self.batch_size))
        self.min_buffer_size = max(raw_min, self.batch_size)

        self._device  = _get_device()
        self._trained = False

        self._online_net: _DQNNetwork | None = None
        self._target_net: _DQNNetwork | None = None
        self._optimizer: optim.Optimizer | None = None

    # ── Network management ────────────────────────────────────────────────────

    def _build_networks(self) -> None:
        q_in = self.encoder_hidden_size if self._use_encoder else self.input_size

        def _make() -> _DQNNetwork:
            enc = (
                _GRUEncoder(self.input_size, self.encoder_hidden_size, self.dropout)
                if self._use_encoder else None
            )
            return _DQNNetwork(enc, _QNetwork(q_in, self.hidden_sizes, self.dropout))

        self._online_net = _make().to(self._device)
        self._target_net = _make().to(self._device)
        self._target_net.load_state_dict(self._online_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.Adam(self._online_net.parameters(), lr=self.lr)

    def _sync_target(self) -> None:
        """Copy online network weights to the target network."""
        self._target_net.load_state_dict(self._online_net.state_dict())

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_tab: np.ndarray,
        returns_1d: np.ndarray,
        labels: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> dict:
        """Train the DQN agent over chronological time-series data.

        Parameters
        ----------
        X_tab:
            Scaled feature matrix.  Accepts:

            * ``(N, seq_len, n_features)`` — full sequence windows.  When the
              GRU encoder is enabled these are passed through the encoder to
              produce the RL state.  When the encoder is disabled the last
              timestep is extracted automatically for backward compatibility.
            * ``(N, n_features)`` — pre-extracted last-timestep features
              (legacy / non-encoder path).

        returns_1d:
            1-step price return r(t) = close[t+1]/close[t] − 1, shape (N,).
        labels:
            Optional true signal labels (0=SELL,1=HOLD,2=BUY) for episode-end
            balanced-accuracy logging.
        feature_names:
            Accepted for API parity with other predictors; not used by DQN.
        """
        # Normalise input: strip to last timestep when encoder is not in use.
        if X_tab.ndim == 3 and not self._use_encoder:
            X_tab = X_tab[:, -1, :]

        self._build_networks()
        buffer = _ReplayBuffer(self.buffer_size)

        N = len(X_tab)
        history: dict = {"episode_rewards": [], "episode_balanced_acc": []}
        total_steps = 0
        total_train_steps = max((N - 1) * self.n_episodes, 1)

        self._online_net.train()

        for episode in range(self.n_episodes):
            position = 0
            total_reward = 0.0
            pred_labels: list[int] = []

            for t in range(N - 1):
                state = X_tab[t].astype(np.float32)
                # state shape: (seq_len, n_features) with encoder, (n_features,) without

                epsilon = max(
                    self.epsilon_end,
                    self.epsilon_start - (self.epsilon_start - self.epsilon_end)
                    * total_steps / total_train_steps,
                )

                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(
                            state, dtype=torch.float32, device=self._device
                        ).unsqueeze(0)
                        # s_t: (1, seq_len, n_features) or (1, n_features) — both
                        # valid for _DQNNetwork.forward()
                        q_vals = self._online_net(s_t).squeeze(0).cpu().numpy()
                    action = int(np.argmax(q_vals))

                if action == 2:
                    new_position = 1
                elif action == 0:
                    new_position = 0
                else:
                    new_position = position

                r      = float(returns_1d[t])
                tc     = self.transaction_cost
                reward = new_position * r - abs(new_position - position) * tc

                next_state = X_tab[t + 1].astype(np.float32)
                done = (t == N - 2)

                buffer.push(state, action, reward, next_state, done)

                position     = new_position
                total_reward += reward
                pred_labels.append(action)
                total_steps  += 1

                if len(buffer) >= self.min_buffer_size:
                    self._update(buffer)

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
        enc_info = f", encoder_hidden={self.encoder_hidden_size}" if self._use_encoder else ""
        logger.info(
            f"DQN training complete — "
            f"architecture={self.hidden_sizes}{enc_info}, buffer={len(buffer)}"
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
        # s / ns shape: (batch, seq_len, n_features) with encoder
        #               (batch, n_features)           without encoder
        # _DQNNetwork.forward() handles both paths transparently.

        q_pred = self._online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

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
        - 3-D ``(N, seq_len, n_features)`` — full sequence; fed through the
          GRU encoder when enabled, otherwise last timestep is extracted.
        - 2-D ``(N, n_features)``          — tabular last-step features.

        Returns ``(N, 3)`` float32 array; column order: SELL / HOLD / BUY.
        """
        if not self._trained or self._online_net is None:
            raise RuntimeError("DQN model has not been trained yet.")

        if X.ndim == 3 and not self._use_encoder:
            X = X[:, -1, :]   # legacy: strip to last timestep

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
                "encoder_hidden_size": self.encoder_hidden_size,
                "dropout":            self.dropout,
                "learning_rate":      self.lr,
                "batch_size":         self.batch_size,
                "buffer_size":        self.buffer_size,
                "target_update_freq": self.target_update_freq,
                "discount_factor":    self.gamma,
                "epsilon_start":      self.epsilon_start,
                "epsilon_end":        self.epsilon_end,
                "n_episodes":         self.n_episodes,
                "transaction_cost":   self.transaction_cost,
                "temperature":        self.temperature,
                "horizon":            self.horizon,
                "min_buffer_size":    self.min_buffer_size,
                "trained":            self._trained,
            },
            path,
        )
        enc_info = f", encoder_hidden={self.encoder_hidden_size}" if self._use_encoder else ""
        logger.info(f"Saved DQN model ({self.hidden_sizes}{enc_info}) to {path}")

    def load(self, path: str | Path) -> None:
        data = torch.load(path, map_location=self._device, weights_only=False)
        self.input_size          = data.get("input_size",          self.input_size)
        self.hidden_sizes        = data.get("hidden_sizes",        self.hidden_sizes)
        self.encoder_hidden_size = data.get("encoder_hidden_size", None)
        self._use_encoder        = self.encoder_hidden_size is not None
        self.dropout             = data.get("dropout",             self.dropout)
        self.lr                  = data.get("learning_rate",       self.lr)
        self.batch_size          = data.get("batch_size",          self.batch_size)
        self.buffer_size         = data.get("buffer_size",         self.buffer_size)
        self.target_update_freq  = data.get("target_update_freq",  self.target_update_freq)
        self.gamma               = data.get("discount_factor",     self.gamma)
        self.epsilon_start       = data.get("epsilon_start",       self.epsilon_start)
        self.epsilon_end         = data.get("epsilon_end",         self.epsilon_end)
        self.n_episodes          = data.get("n_episodes",          self.n_episodes)
        self.transaction_cost    = data.get("transaction_cost",    self.transaction_cost)
        self.temperature         = data.get("temperature",         self.temperature)
        self.horizon             = data.get("horizon",             self.horizon)
        self.min_buffer_size     = data.get("min_buffer_size",     self.min_buffer_size)
        self._trained            = data.get("trained",             False)

        self._build_networks()
        self._online_net.load_state_dict(data["online_state_dict"])
        self._online_net.eval()
        enc_info = f", encoder_hidden={self.encoder_hidden_size}" if self._use_encoder else ""
        logger.info(f"Loaded DQN model ({self.hidden_sizes}{enc_info}) from {path}")
