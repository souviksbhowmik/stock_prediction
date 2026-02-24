"""Weighted ensemble of LSTM and XGBoost models."""

from dataclasses import dataclass

import numpy as np

from stock_prediction.config import get_setting
from stock_prediction.models.lstm_model import LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.ensemble")

SIGNAL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclass
class EnsemblePrediction:
    """Result from ensemble prediction."""

    signal: str  # BUY, HOLD, SELL
    signal_idx: int  # 0=SELL, 1=HOLD, 2=BUY
    confidence: float  # max probability
    probabilities: dict[str, float]  # {SELL: p, HOLD: p, BUY: p}
    lstm_probs: dict[str, float]
    xgboost_probs: dict[str, float]


class EnsembleModel:
    """Weighted average ensemble of LSTM and XGBoost."""

    def __init__(
        self,
        lstm: LSTMPredictor,
        xgboost: XGBoostPredictor,
        lstm_weight: float | None = None,
        xgboost_weight: float | None = None,
    ):
        self.lstm = lstm
        self.xgboost = xgboost
        self.lstm_weight = lstm_weight if lstm_weight is not None else get_setting("models", "ensemble", "lstm_weight", default=0.4)
        self.xgboost_weight = xgboost_weight if xgboost_weight is not None else get_setting("models", "ensemble", "xgboost_weight", default=0.6)

    def predict(
        self, X_seq: np.ndarray, X_tab: np.ndarray
    ) -> list[EnsemblePrediction]:
        """Generate ensemble predictions.

        Args:
            X_seq: LSTM input sequences, shape (N, seq_len, n_features)
            X_tab: XGBoost input tabular, shape (N, n_features)
        """
        lstm_probs = self.lstm.predict_proba(X_seq)
        xgb_probs = self.xgboost.predict_proba(X_tab)

        # Weighted average
        ensemble_probs = (
            self.lstm_weight * lstm_probs + self.xgboost_weight * xgb_probs
        )

        predictions = []
        for i in range(len(ensemble_probs)):
            probs = ensemble_probs[i]
            signal_idx = int(np.argmax(probs))
            signal = SIGNAL_MAP[signal_idx]
            confidence = float(probs[signal_idx])

            predictions.append(
                EnsemblePrediction(
                    signal=signal,
                    signal_idx=signal_idx,
                    confidence=confidence,
                    probabilities={
                        "SELL": float(probs[0]),
                        "HOLD": float(probs[1]),
                        "BUY": float(probs[2]),
                    },
                    lstm_probs={
                        "SELL": float(lstm_probs[i][0]),
                        "HOLD": float(lstm_probs[i][1]),
                        "BUY": float(lstm_probs[i][2]),
                    },
                    xgboost_probs={
                        "SELL": float(xgb_probs[i][0]),
                        "HOLD": float(xgb_probs[i][1]),
                        "BUY": float(xgb_probs[i][2]),
                    },
                )
            )

        return predictions

    def predict_single(
        self, X_seq: np.ndarray, X_tab: np.ndarray
    ) -> EnsemblePrediction:
        """Predict for a single sample."""
        if X_seq.ndim == 2:
            X_seq = X_seq[np.newaxis, ...]
        if X_tab.ndim == 1:
            X_tab = X_tab[np.newaxis, ...]
        return self.predict(X_seq, X_tab)[0]
