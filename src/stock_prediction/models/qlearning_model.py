"""Tabular Q-learning agent for stock trading.

Design notes
------------
State space
    A small, configurable subset of the full feature vector is selected
    (default: RSI, MACD_Histogram, BB_Width, Volume_Ratio, Price_Momentum_5d).
    Each selected feature is independently discretised into *n_bins* quantile
    buckets fitted on training data.  With 5 features and 3 bins the state
    space is 3^5 = 243 cells — tractable for a tabular Q-table.

Action space
    0 = SELL  (close / stay flat)
    1 = HOLD  (keep current position)
    2 = BUY   (open / stay long)

Reward function  (Moody & Saffell 2001; Spooner et al. 2018)
    A position-based P&L reward with transaction costs:

        reward(t) = new_position(t) × r(t) − |Δposition| × tc

    where
        r(t)            = 1-step price return at step t  (from reg_targets[:,0]−1)
        new_position(t) = 1 after BUY, 0 after SELL, unchanged after HOLD
        Δposition       = |new_position − old_position|  (0 or 1)
        tc              = transaction_cost  (default 0.1 %)

    Properties
        • BUY when flat  → immediate cost −tc; future reward tracks the return
        • SELL when long → immediate profit r(t) − tc
        • HOLD           → unrealised gain/loss if long, 0 if flat
        • Repeated invalid actions (BUY when already long) receive the
          holding reward without a cost, discouraging churning

Q-learning update
    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]

    The agent runs *n_episodes* passes over the chronological training data
    with ε-greedy exploration (ε decays each episode).

Input at training / inference
    Uses the **last timestep** of the seq-scaler-normalised sequence window,
    i.e. ``X_seq[:, -1, :]`` — exactly what the Ensemble passes as X_seq.
    No extra data path is required.

Probability output
    Softmax over Q-values with a tunable temperature.
    Unknown states (never visited during training) return a HOLD-biased prior.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import HORIZON_THRESHOLDS
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.qlearning")

# Default features used to build the state.  All must exist in feature_names
# produced by FeaturePipeline.  The list is used in order; missing names are
# silently skipped and the remaining ones are kept.
_DEFAULT_STATE_FEATURES: list[str] = [
    "RSI",
    "MACD_Histogram",
    "BB_Width",
    "Volume_Ratio",
    "Price_Momentum_5d",
]

# Prior Q-values for unseen states: slight HOLD bias so the agent defaults to
# doing nothing rather than random trading.
_UNSEEN_PRIOR = np.array([0.0, 0.01, 0.0], dtype=np.float64)


class QLearningPredictor:
    """Tabular Q-learning agent for stock Buy / Hold / Sell decisions.

    The class follows the same ``train`` / ``predict_proba`` / ``predict`` /
    ``save`` / ``load`` API as all other predictors in this project.

    Parameters
    ----------
    input_size:
        Total number of features in the input vector (used for validation).
    n_bins:
        Number of quantile buckets per state feature.
    learning_rate:
        Q-learning step size α.
    discount_factor:
        Future reward discount γ.
    epsilon_start / epsilon_end / epsilon_decay:
        ε-greedy schedule.  ε_episode = max(ε_end, ε_start × ε_decay^episode).
    n_episodes:
        How many passes over the training time-series.
    transaction_cost:
        Fraction of trade value charged per open / close (e.g. 0.001 = 0.1 %).
    temperature:
        Softmax temperature for converting Q-values to probabilities.
        Lower → more peaked; higher → more uniform.
    horizon:
        Forecast horizon (used only to look up signal thresholds for logging).
    """

    def __init__(
        self,
        input_size: int,
        n_bins: int | None = None,
        learning_rate: float | None = None,
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
            return get_setting("models", "qlearning", key, default=default)

        self.n_bins          = n_bins          if n_bins          is not None else int(_cfg("n_bins", 3))
        self.lr              = learning_rate   if learning_rate   is not None else float(_cfg("learning_rate", 0.1))
        self.gamma           = discount_factor if discount_factor is not None else float(_cfg("discount_factor", 0.95))
        self.epsilon_start   = epsilon_start   if epsilon_start   is not None else float(_cfg("epsilon_start", 1.0))
        self.epsilon_end     = epsilon_end     if epsilon_end     is not None else float(_cfg("epsilon_end", 0.01))
        self.epsilon_decay   = epsilon_decay   if epsilon_decay   is not None else float(_cfg("epsilon_decay", 0.995))
        self.n_episodes      = n_episodes      if n_episodes      is not None else int(_cfg("n_episodes", 10))
        self.transaction_cost = transaction_cost if transaction_cost is not None else float(_cfg("transaction_cost", 0.001))
        self.temperature     = temperature     if temperature     is not None else float(_cfg("temperature", 0.5))
        self.horizon         = horizon         if horizon         is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )

        # Filled during training
        self.q_table: dict[tuple, np.ndarray] = {}
        self.bin_edges: list[np.ndarray] | None = None
        self.state_feature_indices: list[int] = []
        self._feature_names: list[str] = []

        # Horizon thresholds (for logging only)
        self._thresholds: tuple[float, float] = HORIZON_THRESHOLDS.get(
            self.horizon, (0.022, -0.022)
        )

    # ── State machinery ───────────────────────────────────────────────────

    def _resolve_state_features(self, feature_names: list[str]) -> None:
        """Map configured state-feature names → column indices.

        Falls back to first min(5, input_size) columns when no configured
        feature names are found in the supplied list.
        """
        wanted: list[str] = get_setting(
            "models", "qlearning", "state_features", default=_DEFAULT_STATE_FEATURES
        )
        indices = [feature_names.index(f) for f in wanted if f in feature_names]
        if not indices:
            logger.warning(
                "None of the configured state_features found in feature_names; "
                "falling back to first 5 columns."
            )
            indices = list(range(min(5, self.input_size)))
        self.state_feature_indices = indices
        self._feature_names = feature_names
        logger.info(
            f"Q-learning state features ({len(indices)}): "
            + ", ".join(
                feature_names[i] if i < len(feature_names) else str(i)
                for i in indices
            )
        )

    def _fit_bins(self, X: np.ndarray) -> None:
        """Fit per-feature quantile bin edges from training data."""
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = []
        for idx in self.state_feature_indices:
            vals = X[:, idx]
            edges = np.percentile(vals, percentiles)
            edges = np.unique(edges)  # collapse duplicate percentiles
            # Ensure at least two distinct edges so searchsorted is meaningful
            if len(edges) < 2:
                edges = np.array([edges[0] - 1e-6, edges[0] + 1e-6])
            self.bin_edges.append(edges)

    def _get_state(self, x: np.ndarray) -> tuple:
        """Discretise a feature vector into a state tuple."""
        state: list[int] = []
        for feat_idx, edges in zip(self.state_feature_indices, self.bin_edges):
            val = float(x[feat_idx])
            # searchsorted on inner edges (excluding min/max) → bin index 0..n_bins-1
            bin_idx = int(np.searchsorted(edges[1:-1], val))
            state.append(bin_idx)
        return tuple(state)

    def _get_q(self, state: tuple) -> np.ndarray:
        """Return (and lazily initialise) the Q-value array for a state."""
        if state not in self.q_table:
            self.q_table[state] = _UNSEEN_PRIOR.copy()
        return self.q_table[state]

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        X_tab: np.ndarray,           # (N, n_features)  last-timestep features
        returns_1d: np.ndarray,      # (N,)  1-step price return per step
        labels: np.ndarray | None = None,   # (N,)  true labels (0/1/2) for logging
        feature_names: list[str] | None = None,
    ) -> dict:
        """Run Q-learning over the chronological training data.

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
            Column names matching axis-1 of X_tab, used to resolve state features.
        """
        if feature_names is None:
            feature_names = [str(i) for i in range(X_tab.shape[1])]

        self._resolve_state_features(feature_names)
        self._fit_bins(X_tab)

        N = len(X_tab)
        history: dict = {"episode_rewards": [], "episode_balanced_acc": []}

        for episode in range(self.n_episodes):
            epsilon = max(
                self.epsilon_end,
                self.epsilon_start * (self.epsilon_decay ** episode),
            )
            position = 0          # 0 = FLAT, 1 = LONG
            total_reward = 0.0
            pred_labels: list[int] = []

            for t in range(N - 1):
                state = self._get_state(X_tab[t])
                q_vals = self._get_q(state)

                # ε-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = int(np.argmax(q_vals))

                # Resolve new position
                if action == 2:      # BUY
                    new_position = 1
                elif action == 0:    # SELL
                    new_position = 0
                else:                # HOLD
                    new_position = position

                # Position-based reward with transaction cost
                r = float(returns_1d[t])
                tc = self.transaction_cost
                reward = new_position * r - abs(new_position - position) * tc

                # Next state and Q-update
                next_state = self._get_state(X_tab[t + 1])
                next_q = self._get_q(next_state)

                q_vals[action] += self.lr * (
                    reward + self.gamma * float(np.max(next_q)) - q_vals[action]
                )

                position = new_position
                total_reward += reward
                pred_labels.append(action)

            avg_reward = total_reward / max(N - 1, 1)
            history["episode_rewards"].append(avg_reward)

            if labels is not None and len(labels) > 0:
                n = min(len(pred_labels), len(labels) - 1)
                ba = balanced_accuracy_score(labels[:n], pred_labels[:n])
                history["episode_balanced_acc"].append(ba)
                if (episode + 1) % max(1, self.n_episodes // 5) == 0:
                    logger.info(
                        f"QL episode {episode+1}/{self.n_episodes} — "
                        f"ε={epsilon:.3f}, avg_reward={avg_reward:.5f}, "
                        f"balanced_acc={ba:.4f}, states_visited={len(self.q_table)}"
                    )
            else:
                if (episode + 1) % max(1, self.n_episodes // 5) == 0:
                    logger.info(
                        f"QL episode {episode+1}/{self.n_episodes} — "
                        f"ε={epsilon:.3f}, avg_reward={avg_reward:.5f}, "
                        f"states_visited={len(self.q_table)}"
                    )

        logger.info(
            f"Q-learning training complete — "
            f"total states in Q-table: {len(self.q_table)}, "
            f"n_bins={self.n_bins}"
        )
        return history

    # ── Evaluation ────────────────────────────────────────────────────────

    def compute_balanced_accuracy(
        self, X_tab: np.ndarray, labels: np.ndarray
    ) -> float:
        """Balanced accuracy of greedy-policy predictions vs true labels."""
        preds = self.predict(X_tab)
        n = min(len(preds), len(labels))
        return float(balanced_accuracy_score(labels[:n], preds[:n]))

    # ── Prediction ────────────────────────────────────────────────────────

    def _q_values_for_row(self, x: np.ndarray) -> np.ndarray:
        state = self._get_state(x)
        return self._get_q(state).copy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert Q-values to class probabilities via softmax.

        Accepts either:
        - 2-D ``(N, n_features)``   — tabular last-step features
        - 3-D ``(N, T, n_features)`` — sequence; uses the last timestep

        Returns ``(N, 3)`` float32 array; column order: SELL / HOLD / BUY.
        """
        if self.bin_edges is None:
            raise RuntimeError("Model has not been trained yet.")

        if X.ndim == 3:
            X = X[:, -1, :]   # take last timestep from sequence

        probs = np.empty((len(X), 3), dtype=np.float32)
        for i, x in enumerate(X):
            q = self._q_values_for_row(x)
            probs[i] = softmax(q / max(self.temperature, 1e-8)).astype(np.float32)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels 0=SELL / 1=HOLD / 2=BUY."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "q_table":               self.q_table,
                "bin_edges":             self.bin_edges,
                "state_feature_indices": self.state_feature_indices,
                "feature_names":         self._feature_names,
                "input_size":            self.input_size,
                "n_bins":                self.n_bins,
                "learning_rate":         self.lr,
                "discount_factor":       self.gamma,
                "epsilon_start":         self.epsilon_start,
                "epsilon_end":           self.epsilon_end,
                "epsilon_decay":         self.epsilon_decay,
                "n_episodes":            self.n_episodes,
                "transaction_cost":      self.transaction_cost,
                "temperature":           self.temperature,
                "horizon":               self.horizon,
            },
            path,
        )
        logger.info(
            f"Saved Q-table ({len(self.q_table)} states) to {path}"
        )

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self.q_table               = data["q_table"]
        self.bin_edges             = data["bin_edges"]
        self.state_feature_indices = data["state_feature_indices"]
        self._feature_names        = data.get("feature_names", [])
        self.input_size            = data.get("input_size", self.input_size)
        self.n_bins                = data.get("n_bins", self.n_bins)
        self.lr                    = data.get("learning_rate", self.lr)
        self.gamma                 = data.get("discount_factor", self.gamma)
        self.epsilon_start         = data.get("epsilon_start", self.epsilon_start)
        self.epsilon_end           = data.get("epsilon_end", self.epsilon_end)
        self.epsilon_decay         = data.get("epsilon_decay", self.epsilon_decay)
        self.n_episodes            = data.get("n_episodes", self.n_episodes)
        self.transaction_cost      = data.get("transaction_cost", self.transaction_cost)
        self.temperature           = data.get("temperature", self.temperature)
        self.horizon               = data.get("horizon", self.horizon)
        logger.info(
            f"Loaded Q-table ({len(self.q_table)} states) from {path}"
        )
