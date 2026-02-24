"""Training orchestration for stock prediction models."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import FeaturePipeline
from stock_prediction.models.lstm_model import LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.models.ensemble import EnsembleModel
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.trainer")

# ---------------------------------------------------------------------------
# Hyperparameter search grids
# XGBoost: tune max_depth and learning_rate (most impactful); subsample and
# colsample_bytree remain at their config defaults.
# ---------------------------------------------------------------------------
_XGB_PARAM_GRID: list[dict] = [
    {"max_depth": 3, "learning_rate": 0.10},
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.10},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.10},
    {"max_depth": 8, "learning_rate": 0.05},
]

# LSTM: tune hidden_size, dropout, and learning_rate.
_LSTM_PARAM_GRID: list[dict] = [
    {"hidden_size": 64,  "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 64,  "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.0005},
    {"hidden_size": 256, "dropout": 0.3, "learning_rate": 0.001},
]


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

        Pipeline:
          1. Build features and do a chronological 80/20 split.
          2. Grid-search hyperparameters for XGBoost and LSTM using the val split.
          3. Report ensemble val accuracy from the best combination.
          4. Refit scalers on the *full* dataset.
          5. Retrain final XGBoost and LSTM on full data with the best hyperparams
             (n_estimators / epochs scaled up proportionally to the larger dataset).
          6. Save the full-data models and scalers.

        Returns (EnsembleModel, validation_accuracy) on success,
        or (None, None) if there is no training data.
        """
        logger.info(f"Training models for {symbol}")

        # ── 1. Build features ─────────────────────────────────────────────
        sequences, tabular, labels, feature_names = self.pipeline.prepare_training_data(
            symbol, start_date, end_date
        )

        if len(sequences) == 0:
            return None, None

        n_samples, seq_len, n_features = sequences.shape

        # ── 2. Chronological train / val split ────────────────────────────
        split_idx = int(n_samples * self.train_split)
        X_seq_train, X_seq_val = sequences[:split_idx], sequences[split_idx:]
        X_tab_train, X_tab_val = tabular[:split_idx], tabular[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]

        # Fit scalers on train split only (no leakage during tuning)
        scaler = StandardScaler()
        X_tab_train_s = scaler.fit_transform(X_tab_train)
        X_tab_val_s = scaler.transform(X_tab_val)

        seq_scaler = StandardScaler()
        X_seq_train_s = seq_scaler.fit_transform(
            X_seq_train.reshape(-1, n_features)
        ).reshape(split_idx, seq_len, n_features)
        X_seq_val_s = seq_scaler.transform(
            X_seq_val.reshape(-1, n_features)
        ).reshape(len(X_seq_val), seq_len, n_features)

        # ── 3. Compute class weights from training split ───────────────────
        classes = np.array([0, 1, 2])
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        sample_weights_train = compute_sample_weight("balanced", y=y_train)
        logger.info(
            f"Class weights for {symbol} (SELL/HOLD/BUY): "
            + "/".join(f"{w:.2f}" for w in class_weights)
        )

        # ── 4. Hyperparameter tuning on val split ─────────────────────────
        logger.info(f"Tuning XGBoost hyperparameters for {symbol}...")
        best_xgb_params, best_n_estimators, xgb_val_acc = self._tune_xgboost(
            X_tab_train_s, y_train, X_tab_val_s, y_val,
            sample_weight=sample_weights_train,
        )

        logger.info(f"Tuning LSTM hyperparameters for {symbol}...")
        best_lstm_params, best_lstm_epochs, lstm_val_acc = self._tune_lstm(
            X_seq_train_s, y_train, X_seq_val_s, y_val, n_features,
            class_weights=class_weights,
        )

        # Derive per-stock dynamic ensemble weights from individual val accuracies
        total_acc = lstm_val_acc + xgb_val_acc
        if total_acc > 0:
            lstm_weight = lstm_val_acc / total_acc
            xgb_weight = xgb_val_acc / total_acc
        else:
            lstm_weight = get_setting("models", "ensemble", "lstm_weight", default=0.4)
            xgb_weight = get_setting("models", "ensemble", "xgboost_weight", default=0.6)
        logger.info(
            f"Dynamic ensemble weights for {symbol}: "
            f"LSTM={lstm_weight:.3f} (balanced_acc={lstm_val_acc:.4f}), "
            f"XGB={xgb_weight:.3f} (balanced_acc={xgb_val_acc:.4f})"
        )

        # ── 5. Compute ensemble val accuracy with the best individual models
        xgb_val = XGBoostPredictor(**best_xgb_params, n_estimators=best_n_estimators, early_stopping_rounds=None)
        xgb_val.train(X_tab_train_s, y_train, feature_names=feature_names,
                      sample_weight=sample_weights_train)

        lstm_val = LSTMPredictor(input_size=n_features, **best_lstm_params, epochs=best_lstm_epochs)
        lstm_val.train(X_seq_train_s, y_train, class_weights=class_weights)

        ensemble_val = EnsembleModel(lstm_val, xgb_val, lstm_weight=lstm_weight, xgboost_weight=xgb_weight)
        val_preds = ensemble_val.predict(X_seq_val_s, X_tab_val_s)
        val_pred_labels = np.array([p.signal_idx for p in val_preds])
        val_accuracy = balanced_accuracy_score(y_val, val_pred_labels)
        logger.info(f"Best ensemble balanced val accuracy for {symbol}: {val_accuracy:.4f}")

        # ── 5. Retrain on the full dataset with best hyperparameters ──────
        logger.info(f"Retraining on full dataset for {symbol}...")

        # Refit scalers on ALL data (no leakage risk — predictions use these scalers)
        full_scaler = StandardScaler()
        X_tab_full_s = full_scaler.fit_transform(tabular)

        full_seq_scaler = StandardScaler()
        X_seq_full_s = full_seq_scaler.fit_transform(
            sequences.reshape(-1, n_features)
        ).reshape(n_samples, seq_len, n_features)

        # Recompute class weights on the full dataset for final models
        class_weights_full = compute_class_weight("balanced", classes=classes, y=labels)
        sample_weights_full = compute_sample_weight("balanced", y=labels)

        # XGBoost: fixed n_estimators (no early stopping — no held-out set)
        # Scale up proportionally since full data ~1/train_split times larger.
        n_est_full = max(best_n_estimators, int(best_n_estimators / self.train_split))
        xgb_final = XGBoostPredictor(
            **best_xgb_params,
            n_estimators=n_est_full,
            early_stopping_rounds=None,
        )
        xgb_final.train(X_tab_full_s, labels, feature_names=feature_names,
                        sample_weight=sample_weights_full)

        # LSTM: fixed epoch count (no early stopping)
        epochs_full = max(best_lstm_epochs, int(best_lstm_epochs / self.train_split))
        lstm_final = LSTMPredictor(
            input_size=n_features, **best_lstm_params, epochs=epochs_full
        )
        lstm_final.train(X_seq_full_s, labels, class_weights=class_weights_full)

        ensemble = EnsembleModel(lstm_final, xgb_final, lstm_weight=lstm_weight, xgboost_weight=xgb_weight)

        # ── 6. Save full-data models and scalers ──────────────────────────
        self._save_models(
            symbol, lstm_final, xgb_final, full_scaler, full_seq_scaler,
            feature_names, lstm_weight, xgb_weight,
        )

        return ensemble, val_accuracy

    # ── Hyperparameter tuning helpers ─────────────────────────────────────

    def _tune_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[dict, int, float]:
        """Grid search over _XGB_PARAM_GRID; returns (best_params, best_n_estimators, best_val_acc)."""
        best_acc = -1.0
        best_params: dict = {}
        best_n_est = 500

        for params in _XGB_PARAM_GRID:
            model = XGBoostPredictor(**params)
            model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
            acc = balanced_accuracy_score(y_val, model.predict(X_val))
            n_est = getattr(model.model, "best_iteration", self._cfg_xgb_n_estimators()) + 1
            logger.info(f"  XGB {params} → balanced_acc={acc:.4f}, best_iter={n_est}")
            if acc > best_acc:
                best_acc = acc
                best_params = dict(params)
                best_n_est = n_est

        logger.info(
            f"Best XGB: {best_params}, n_estimators={best_n_est}, balanced_acc={best_acc:.4f}"
        )
        return best_params, best_n_est, best_acc

    def _tune_lstm(
        self,
        X_seq_train: np.ndarray,
        y_train: np.ndarray,
        X_seq_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int,
        class_weights: np.ndarray | None = None,
    ) -> tuple[dict, int, float]:
        """Grid search over _LSTM_PARAM_GRID; returns (best_params, best_epochs, best_val_acc)."""
        best_acc = -1.0
        best_params: dict = {}
        best_epochs = get_setting("models", "lstm", "epochs", default=50)

        for params in _LSTM_PARAM_GRID:
            model = LSTMPredictor(input_size=n_features, **params)
            history = model.train(X_seq_train, y_train, X_seq_val, y_val,
                                  class_weights=class_weights)
            acc = balanced_accuracy_score(y_val, model.predict(X_seq_val))
            epoch = int(history.get("best_epoch", model.epochs))  # type: ignore[arg-type]
            logger.info(f"  LSTM {params} → balanced_acc={acc:.4f}, best_epoch={epoch}")
            if acc > best_acc:
                best_acc = acc
                best_params = dict(params)
                best_epochs = epoch

        logger.info(
            f"Best LSTM: {best_params}, epochs={best_epochs}, balanced_acc={best_acc:.4f}"
        )
        return best_params, best_epochs, best_acc

    def _cfg_xgb_n_estimators(self) -> int:
        return int(get_setting("models", "xgboost", "n_estimators", default=500))

    # ── Batch training ────────────────────────────────────────────────────

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
                results[symbol] = {
                    "status": "success",
                    "reason": "",
                    "accuracy": accuracy,
                    "model": model,
                }
            except ValueError as e:
                # Data availability / insufficiency problems
                logger.error(f"No usable data for {symbol}: {e}")
                results[symbol] = {
                    "status": "no_data",
                    "reason": str(e),
                    "accuracy": None,
                    "model": None,
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

    # ── Model persistence ─────────────────────────────────────────────────

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

        # Restore per-stock dynamic weights; fall back to config if loading an
        # older model that was saved before dynamic weighting was introduced.
        lstm_weight = meta.get("lstm_weight")
        xgb_weight = meta.get("xgb_weight")
        ensemble = EnsembleModel(lstm, xgb, lstm_weight=lstm_weight, xgboost_weight=xgb_weight)

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
        lstm_weight: float,
        xgb_weight: float,
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
            "lstm_weight": lstm_weight,
            "xgb_weight": xgb_weight,
            "trained_at": datetime.now().isoformat(),
        }, model_dir / "meta.joblib")

        logger.info(f"Saved models for {symbol} to {model_dir}")
