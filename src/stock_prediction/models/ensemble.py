"""Weighted ensemble of LSTM, XGBoost, Encoder-Decoder, Prophet, TFT, and Q-learning models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stock_prediction.config import get_setting
from stock_prediction.models.lstm_model import LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.ensemble")

SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Lazy imports for new model types — avoids hard dependency if not selected
def _get_ed_type():
    from stock_prediction.models.encoder_decoder_model import EncoderDecoderPredictor
    return EncoderDecoderPredictor

def _get_prophet_type():
    from stock_prediction.models.prophet_model import ProphetPredictor
    return ProphetPredictor

def _get_tft_type():
    from stock_prediction.models.tft_model import TFTPredictor
    return TFTPredictor

def _get_ql_type():
    from stock_prediction.models.qlearning_model import QLearningPredictor
    return QLearningPredictor

def _get_dqn_type():
    from stock_prediction.models.dqn_model import DQNPredictor
    return DQNPredictor


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""

    signal: str          # BUY, HOLD, SELL
    signal_idx: int      # 0=SELL, 1=HOLD, 2=BUY
    confidence: float    # max probability
    probabilities: dict[str, float]        # {SELL: p, HOLD: p, BUY: p}
    lstm_probs: dict[str, float]
    xgboost_probs: dict[str, float]
    encoder_decoder_probs: dict[str, float] = field(
        default_factory=lambda: {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    )
    prophet_probs: dict[str, float] = field(
        default_factory=lambda: {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    )
    tft_probs: dict[str, float] = field(
        default_factory=lambda: {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    )
    qlearning_probs: dict[str, float] = field(
        default_factory=lambda: {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    )
    dqn_probs: dict[str, float] = field(
        default_factory=lambda: {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
    )


class EnsembleModel:
    """Weighted-average ensemble of up to seven model types.

    Supported models (any combination):
      - lstm           : LSTM sequence classifier
      - xgboost        : XGBoost tabular classifier
      - encoder_decoder: Encoder-Decoder LSTM regressor (ratios → probs via Gaussian CDF)
      - prophet        : Prophet time-series regressor (single forecast broadcast)
      - tft            : Temporal Fusion Transformer regressor (ratios → probs via Gaussian CDF)
      - qlearning      : Tabular Q-learning agent (Q-values → softmax probs)
      - dqn            : Deep Q-Network agent (Q-values → softmax probs)

    At least one model must be provided.  The weights must sum to 1.0; the
    caller (ModelTrainer) derives them dynamically from validation balanced
    accuracies.
    """

    def __init__(
        self,
        lstm=None,
        xgboost=None,
        encoder_decoder=None,
        prophet=None,
        tft=None,
        qlearning=None,
        dqn=None,
        lstm_weight: float | None = None,
        xgboost_weight: float | None = None,
        encoder_decoder_weight: float | None = None,
        prophet_weight: float | None = None,
        tft_weight: float | None = None,
        qlearning_weight: float | None = None,
        dqn_weight: float | None = None,
    ):
        if (lstm is None and xgboost is None and encoder_decoder is None
                and prophet is None and tft is None and qlearning is None
                and dqn is None):
            raise ValueError(
                "At least one of lstm / xgboost / encoder_decoder / prophet / tft / "
                "qlearning / dqn must be provided"
            )

        self.lstm = lstm
        self.xgboost = xgboost
        self.encoder_decoder = encoder_decoder
        self.prophet = prophet
        self.tft = tft
        self.qlearning = qlearning
        self.dqn = dqn

        self.lstm_weight = lstm_weight if lstm_weight is not None else (
            get_setting("models", "ensemble", "lstm_weight", default=0.4)
        )
        self.xgboost_weight = xgboost_weight if xgboost_weight is not None else (
            get_setting("models", "ensemble", "xgboost_weight", default=0.6)
        )
        self.encoder_decoder_weight = encoder_decoder_weight if encoder_decoder_weight is not None else 0.0
        self.prophet_weight = prophet_weight if prophet_weight is not None else 0.0
        self.tft_weight = tft_weight if tft_weight is not None else 0.0
        self.qlearning_weight = qlearning_weight if qlearning_weight is not None else 0.0
        self.dqn_weight = dqn_weight if dqn_weight is not None else 0.0

    def predict(
        self, X_seq: np.ndarray | None, X_tab: np.ndarray | None
    ) -> list[EnsemblePrediction]:
        """Generate predictions for N samples.

        Args:
            X_seq: (N, seq_len, n_features) — used by lstm, encoder_decoder, and tft.
                   May be None if none of those models are active.
            X_tab: (N, n_features) — used by xgboost.
                   May be None if xgboost is not active.
        """
        # Determine N
        if X_seq is not None:
            N = X_seq.shape[0]
        elif X_tab is not None:
            N = X_tab.shape[0]
        else:
            raise ValueError("X_seq and X_tab cannot both be None")

        zeros = np.zeros((N, 3), dtype=np.float32)

        # Collect per-model probability arrays and weights
        weighted_sum = np.zeros((N, 3), dtype=np.float32)
        total_weight = 0.0

        lstm_probs    = zeros.copy()
        xgb_probs     = zeros.copy()
        ed_probs      = zeros.copy()
        prophet_probs = zeros.copy()
        tft_probs     = zeros.copy()
        ql_probs      = zeros.copy()
        dqn_probs     = zeros.copy()

        if self.lstm is not None and X_seq is not None:
            lstm_probs = self.lstm.predict_proba(X_seq)
            weighted_sum += self.lstm_weight * lstm_probs
            total_weight += self.lstm_weight

        if self.xgboost is not None and X_tab is not None:
            xgb_probs = self.xgboost.predict_proba(X_tab)
            weighted_sum += self.xgboost_weight * xgb_probs
            total_weight += self.xgboost_weight

        if self.encoder_decoder is not None and X_seq is not None:
            ed_probs = self.encoder_decoder.predict_proba(X_seq)
            weighted_sum += self.encoder_decoder_weight * ed_probs
            total_weight += self.encoder_decoder_weight

        if self.prophet is not None:
            prophet_probs = self.prophet.predict_proba(N)
            weighted_sum += self.prophet_weight * prophet_probs
            total_weight += self.prophet_weight

        if self.tft is not None and X_seq is not None:
            tft_probs = self.tft.predict_proba(X_seq)
            weighted_sum += self.tft_weight * tft_probs
            total_weight += self.tft_weight

        if self.qlearning is not None and X_seq is not None:
            # Q-learning uses last timestep of the sequence for state lookup
            ql_probs = self.qlearning.predict_proba(X_seq)
            weighted_sum += self.qlearning_weight * ql_probs
            total_weight += self.qlearning_weight

        if self.dqn is not None and X_seq is not None:
            # DQN uses last timestep of the sequence as continuous state
            dqn_probs = self.dqn.predict_proba(X_seq)
            weighted_sum += self.dqn_weight * dqn_probs
            total_weight += self.dqn_weight

        ensemble_probs = weighted_sum / max(total_weight, 1e-8)

        predictions = []
        for i in range(N):
            probs = ensemble_probs[i]
            signal_idx = int(np.argmax(probs))
            predictions.append(
                EnsemblePrediction(
                    signal=SIGNAL_MAP[signal_idx],
                    signal_idx=signal_idx,
                    confidence=float(probs[signal_idx]),
                    probabilities={
                        "SELL": float(probs[0]),
                        "HOLD": float(probs[1]),
                        "BUY":  float(probs[2]),
                    },
                    lstm_probs={
                        "SELL": float(lstm_probs[i][0]),
                        "HOLD": float(lstm_probs[i][1]),
                        "BUY":  float(lstm_probs[i][2]),
                    },
                    xgboost_probs={
                        "SELL": float(xgb_probs[i][0]),
                        "HOLD": float(xgb_probs[i][1]),
                        "BUY":  float(xgb_probs[i][2]),
                    },
                    encoder_decoder_probs={
                        "SELL": float(ed_probs[i][0]),
                        "HOLD": float(ed_probs[i][1]),
                        "BUY":  float(ed_probs[i][2]),
                    },
                    prophet_probs={
                        "SELL": float(prophet_probs[i][0]),
                        "HOLD": float(prophet_probs[i][1]),
                        "BUY":  float(prophet_probs[i][2]),
                    },
                    tft_probs={
                        "SELL": float(tft_probs[i][0]),
                        "HOLD": float(tft_probs[i][1]),
                        "BUY":  float(tft_probs[i][2]),
                    },
                    qlearning_probs={
                        "SELL": float(ql_probs[i][0]),
                        "HOLD": float(ql_probs[i][1]),
                        "BUY":  float(ql_probs[i][2]),
                    },
                    dqn_probs={
                        "SELL": float(dqn_probs[i][0]),
                        "HOLD": float(dqn_probs[i][1]),
                        "BUY":  float(dqn_probs[i][2]),
                    },
                )
            )

        return predictions

    def predict_single(
        self, X_seq: np.ndarray | None, X_tab: np.ndarray | None
    ) -> EnsemblePrediction:
        """Predict for a single sample."""
        if X_seq is not None and X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]
        if X_tab is not None and X_tab.ndim == 1:
            X_tab = X_tab[np.newaxis, ...]
        return self.predict(X_seq, X_tab)[0]
