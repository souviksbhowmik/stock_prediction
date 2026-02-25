"""Weighted ensemble of LSTM, XGBoost, Encoder-Decoder, and Prophet models."""

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


class EnsembleModel:
    """Weighted-average ensemble of up to four model types.

    Supported models (any combination):
      - lstm           : LSTM sequence classifier
      - xgboost        : XGBoost tabular classifier
      - encoder_decoder: Encoder-Decoder LSTM regressor (ratios → probs via Gaussian CDF)
      - prophet        : Prophet time-series regressor (single forecast broadcast)

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
        lstm_weight: float | None = None,
        xgboost_weight: float | None = None,
        encoder_decoder_weight: float | None = None,
        prophet_weight: float | None = None,
    ):
        if lstm is None and xgboost is None and encoder_decoder is None and prophet is None:
            raise ValueError("At least one of lstm / xgboost / encoder_decoder / prophet must be provided")

        self.lstm = lstm
        self.xgboost = xgboost
        self.encoder_decoder = encoder_decoder
        self.prophet = prophet

        self.lstm_weight = lstm_weight if lstm_weight is not None else (
            get_setting("models", "ensemble", "lstm_weight", default=0.4)
        )
        self.xgboost_weight = xgboost_weight if xgboost_weight is not None else (
            get_setting("models", "ensemble", "xgboost_weight", default=0.6)
        )
        self.encoder_decoder_weight = encoder_decoder_weight if encoder_decoder_weight is not None else 0.0
        self.prophet_weight = prophet_weight if prophet_weight is not None else 0.0

    def predict(
        self, X_seq: np.ndarray | None, X_tab: np.ndarray | None
    ) -> list[EnsemblePrediction]:
        """Generate predictions for N samples.

        Args:
            X_seq: (N, seq_len, n_features) — used by lstm and encoder_decoder.
                   May be None if neither model is active.
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

        lstm_probs = zeros.copy()
        xgb_probs = zeros.copy()
        ed_probs = zeros.copy()
        prophet_probs = zeros.copy()

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
