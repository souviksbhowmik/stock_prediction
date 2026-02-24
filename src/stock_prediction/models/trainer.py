"""Training orchestration for stock prediction models."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import FeaturePipeline
from stock_prediction.models.lstm_model import LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.models.ensemble import EnsembleModel
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.trainer")


class ModelTrainer:
    """Train LSTM + XGBoost models for each stock."""

    def __init__(self, use_news: bool = True, use_llm: bool = True):
        self.pipeline = FeaturePipeline(use_news=use_news, use_llm=use_llm)
        self.save_dir = Path(get_setting("models", "save_dir", default="data/models"))
        self.train_split = get_setting("models", "train_split", default=0.8)

    def train_stock(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[EnsembleModel | None, float | None]:
        """Train models for a single stock.

        Returns (EnsembleModel, validation_accuracy) on success,
        or (None, None) if there is no training data.
        """
        logger.info(f"Training models for {symbol}")

        # Build features
        sequences, tabular, labels, feature_names = self.pipeline.prepare_training_data(
            symbol, start_date, end_date
        )

        if len(sequences) == 0:
            logger.error(f"No training data for {symbol}")
            return None, None

        # Chronological train/val split
        split_idx = int(len(sequences) * self.train_split)
        X_seq_train, X_seq_val = sequences[:split_idx], sequences[split_idx:]
        X_tab_train, X_tab_val = tabular[:split_idx], tabular[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        # Scale tabular features
        scaler = StandardScaler()
        X_tab_train_scaled = scaler.fit_transform(X_tab_train)
        X_tab_val_scaled = scaler.transform(X_tab_val)

        # Scale sequences (reshape, scale, reshape back)
        n_samples, seq_len, n_features = X_seq_train.shape
        seq_scaler = StandardScaler()
        X_seq_train_flat = X_seq_train.reshape(-1, n_features)
        X_seq_train_scaled = seq_scaler.fit_transform(X_seq_train_flat).reshape(
            n_samples, seq_len, n_features
        )
        X_seq_val_scaled = seq_scaler.transform(
            X_seq_val.reshape(-1, n_features)
        ).reshape(len(X_seq_val), seq_len, n_features)

        # Train LSTM
        logger.info(f"Training LSTM for {symbol}...")
        lstm = LSTMPredictor(input_size=n_features)
        lstm.train(X_seq_train_scaled, y_train, X_seq_val_scaled, y_val)

        # Train XGBoost
        logger.info(f"Training XGBoost for {symbol}...")
        xgb = XGBoostPredictor()
        xgb.train(X_tab_train_scaled, y_train, X_tab_val_scaled, y_val, feature_names)

        # Create ensemble
        ensemble = EnsembleModel(lstm, xgb)

        # Evaluate on validation set
        predictions = ensemble.predict(X_seq_val_scaled, X_tab_val_scaled)
        pred_labels = np.array([p.signal_idx for p in predictions])
        accuracy = np.mean(pred_labels == y_val)
        logger.info(f"Validation accuracy for {symbol}: {accuracy:.4f}")

        # Save models
        self._save_models(symbol, lstm, xgb, scaler, seq_scaler, feature_names)

        return ensemble, float(accuracy)

    def train_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict]:
        """Train models for multiple stocks.

        Returns a dict mapping symbol to a result dict with keys:
          status   : 'success' | 'no_data' | 'failed'
          reason   : human-readable failure reason (empty string on success)
          accuracy : validation accuracy float, or None on failure
          model    : EnsembleModel or None
        """
        results: dict[str, dict] = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"Training {i + 1}/{len(symbols)}: {symbol}")
            try:
                model, accuracy = self.train_stock(symbol, start_date, end_date)
                if model is None:
                    results[symbol] = {
                        "status": "no_data",
                        "reason": "No training data returned by feature pipeline",
                        "accuracy": None,
                        "model": None,
                    }
                else:
                    results[symbol] = {
                        "status": "success",
                        "reason": "",
                        "accuracy": accuracy,
                        "model": model,
                    }
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                results[symbol] = {
                    "status": "failed",
                    "reason": str(e),
                    "accuracy": None,
                    "model": None,
                }
        return results

    def load_models(self, symbol: str) -> tuple[EnsembleModel, StandardScaler, StandardScaler, int | None]:
        """Load trained models for a symbol.

        Returns (ensemble, scaler, seq_scaler, model_age_days).
        model_age_days is None if the model has no trained_at timestamp.
        """
        model_dir = self.save_dir / symbol.replace(".", "_")

        lstm_path = model_dir / "lstm.pt"
        xgb_path = model_dir / "xgboost.joblib"
        meta_path = model_dir / "meta.joblib"

        if not all(p.exists() for p in [lstm_path, xgb_path, meta_path]):
            raise FileNotFoundError(f"Models not found for {symbol} in {model_dir}")

        meta = joblib.load(meta_path)

        lstm = LSTMPredictor(input_size=meta["input_size"])
        lstm.load(lstm_path)

        xgb = XGBoostPredictor()
        xgb.load(xgb_path)

        ensemble = EnsembleModel(lstm, xgb)

        model_age_days = None
        trained_at = meta.get("trained_at")
        if trained_at:
            trained_dt = datetime.fromisoformat(trained_at)
            model_age_days = (datetime.now() - trained_dt).days

        return ensemble, meta["scaler"], meta["seq_scaler"], model_age_days

    def _save_models(
        self,
        symbol: str,
        lstm: LSTMPredictor,
        xgb: XGBoostPredictor,
        scaler: StandardScaler,
        seq_scaler: StandardScaler,
        feature_names: list[str],
    ) -> None:
        model_dir = self.save_dir / symbol.replace(".", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        lstm.save(model_dir / "lstm.pt")
        xgb.save(model_dir / "xgboost.joblib")
        joblib.dump({
            "scaler": scaler,
            "seq_scaler": seq_scaler,
            "feature_names": feature_names,
            "input_size": lstm.input_size,
            "trained_at": datetime.now().isoformat(),
        }, model_dir / "meta.joblib")

        logger.info(f"Saved models for {symbol} to {model_dir}")
