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
from stock_prediction.models.encoder_decoder_model import EncoderDecoderPredictor
from stock_prediction.models.prophet_model import ProphetPredictor
from stock_prediction.models.ensemble import EnsembleModel
from stock_prediction.utils.logging import get_logger
from stock_prediction.utils.plot_utils import generate_plots

logger = get_logger("models.trainer")

# Models that can be selected for training.  Add new model IDs here as they
# are implemented.
AVAILABLE_MODELS: list[str] = ["lstm", "xgboost", "encoder_decoder", "prophet"]

# ---------------------------------------------------------------------------
# Hyperparameter search grids
# ---------------------------------------------------------------------------
_XGB_PARAM_GRID: list[dict] = [
    {"max_depth": 3, "learning_rate": 0.10},
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 4, "learning_rate": 0.10},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.10},
    {"max_depth": 8, "learning_rate": 0.05},
]

_LSTM_PARAM_GRID: list[dict] = [
    {"hidden_size": 64,  "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 64,  "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.0005},
    {"hidden_size": 256, "dropout": 0.3, "learning_rate": 0.001},
]

_ED_PARAM_GRID: list[dict] = [
    {"hidden_size": 64,  "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 64,  "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.3, "learning_rate": 0.001},
    {"hidden_size": 128, "dropout": 0.2, "learning_rate": 0.0005},
    {"hidden_size": 256, "dropout": 0.3, "learning_rate": 0.001},
]


class ModelTrainer:
    """Train and persist prediction models for each stock."""

    def __init__(
        self,
        use_news: bool = True,
        use_llm: bool = True,
        use_financials: bool = True,
    ):
        self.pipeline = FeaturePipeline(
            use_news=use_news, use_llm=use_llm, use_financials=use_financials
        )
        self.save_dir = Path(get_setting("models", "save_dir", default="data/models"))
        self.train_split = get_setting("models", "train_split", default=0.8)

    # =========================================================================
    # Public API
    # =========================================================================

    def train_stock(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        selected_models: list[str] | None = None,
    ) -> tuple[EnsembleModel | None, float | None]:
        """Train models for a single stock.

        Pipeline
        --------
        1. Build features; chronological 80/20 split.
        2. Grid-search hyperparameters for selected model(s) on the val split.
        3. Derive ensemble weights from each model's validation balanced accuracy.
        4. Refit scalers on full data; retrain all selected models at full scale.
        5. Save models and meta.

        Returns ``(EnsembleModel, val_balanced_accuracy)`` on success or
        ``(None, None)`` when no usable training data exists.
        """
        logger.info(f"Training models for {symbol}")

        if selected_models is None:
            selected_models = list(get_setting("models", "selected_models", default=["lstm"]))
        selected_models = [m.strip().lower() for m in selected_models]
        invalid = [m for m in selected_models if m not in AVAILABLE_MODELS]
        if invalid:
            raise ValueError(f"Unknown model(s): {invalid}. Available: {AVAILABLE_MODELS}")
        if not selected_models:
            raise ValueError("selected_models must contain at least one model.")

        use_lstm          = "lstm"           in selected_models
        use_xgboost       = "xgboost"        in selected_models
        use_encoder_decoder = "encoder_decoder" in selected_models
        use_prophet       = "prophet"        in selected_models
        use_sequence      = use_lstm or use_encoder_decoder
        logger.info(f"Selected models for {symbol}: {selected_models}")

        horizon = int(get_setting("features", "prediction_horizon", default=1))

        # ── 1a. Classification data (LSTM / XGBoost baseline) ────────────
        # Always built when any sequence/tabular model is selected; also used
        # to derive scalers for the encoder-decoder input.
        sequences: np.ndarray | None = None
        tabular: np.ndarray | None = None
        labels: np.ndarray | None = None
        feature_names: list[str] = []

        if use_lstm or use_xgboost or use_encoder_decoder:
            sequences, tabular, labels, feature_names = (
                self.pipeline.prepare_training_data(symbol, start_date, end_date)
            )
            if len(sequences) == 0:
                return None, None
            n_samples, seq_len, n_features = sequences.shape
        elif use_prophet:
            # Prophet-only path: we still need feature_names + n_features for meta.
            # Call prepare_training_data just to get them (cheap if cached).
            try:
                sequences, tabular, labels, feature_names = (
                    self.pipeline.prepare_training_data(symbol, start_date, end_date)
                )
                if len(sequences) == 0:
                    return None, None
                n_samples, seq_len, n_features = sequences.shape
            except Exception:
                n_samples, seq_len, n_features = 0, 60, 0

        # ── 1b. Regression data (Encoder-Decoder) ────────────────────────
        seq_reg: np.ndarray | None = None
        reg_targets: np.ndarray | None = None
        labels_reg: np.ndarray | None = None
        n_reg = 0
        if use_encoder_decoder:
            seq_reg, _, reg_targets, labels_reg, _ = (
                self.pipeline.prepare_regression_data(symbol, start_date, end_date)
            )
            n_reg = len(seq_reg)
            if n_reg == 0:
                logger.warning(f"No regression data for {symbol}, skipping encoder_decoder")
                use_encoder_decoder = False
                selected_models = [m for m in selected_models if m != "encoder_decoder"]

        # ── 1c. Prophet + plot data ──────────────────────────────────────
        # prepare_prophet_data is always called — it provides (dates, close)
        # needed for plot generation regardless of whether Prophet is selected.
        dates_all: np.ndarray | None = None
        close_all: np.ndarray | None = None
        prophet_feature_df = None
        try:
            dates_all, close_all, prophet_feature_df, _ = self.pipeline.prepare_prophet_data(
                symbol, start_date, end_date
            )
        except Exception as e:
            logger.warning(f"Could not fetch dates/close for {symbol}: {e}")
        if not use_prophet:
            prophet_feature_df = None  # don't use features if prophet not selected

        # ── 2. Train/val split ───────────────────────────────────────────
        split_idx = int(n_samples * self.train_split) if n_samples > 0 else 0

        # Scalers fitted on training split only (no leakage)
        scaler, seq_scaler = StandardScaler(), StandardScaler()

        if tabular is not None and sequences is not None:
            X_tab_train = tabular[:split_idx]
            X_tab_val   = tabular[split_idx:]
            X_tab_train_s = scaler.fit_transform(X_tab_train)
            X_tab_val_s   = scaler.transform(X_tab_val)

            X_seq_train_s = seq_scaler.fit_transform(
                sequences[:split_idx].reshape(-1, n_features)
            ).reshape(split_idx, seq_len, n_features)
            X_seq_val_s = seq_scaler.transform(
                sequences[split_idx:].reshape(-1, n_features)
            ).reshape(n_samples - split_idx, seq_len, n_features)

            y_train = labels[:split_idx]
            y_val   = labels[split_idx:]
        else:
            X_tab_train_s = X_tab_val_s = np.zeros((0, n_features))
            X_seq_train_s = X_seq_val_s = np.zeros((0, seq_len if seq_len else 60, n_features))
            y_train = y_val = np.zeros(0, dtype=np.int64)

        # Regression split (encoder-decoder)
        X_seq_reg_train_s: np.ndarray | None = None
        X_seq_reg_val_s: np.ndarray | None = None
        y_reg_train: np.ndarray | None = None
        y_reg_val: np.ndarray | None = None
        split_idx_reg = 0

        if use_encoder_decoder and seq_reg is not None and reg_targets is not None:
            split_idx_reg = int(n_reg * self.train_split)
            X_seq_reg_train_s = seq_scaler.transform(
                seq_reg[:split_idx_reg].reshape(-1, n_features)
            ).reshape(split_idx_reg, seq_len, n_features)
            X_seq_reg_val_s = seq_scaler.transform(
                seq_reg[split_idx_reg:].reshape(-1, n_features)
            ).reshape(n_reg - split_idx_reg, seq_len, n_features)
            y_reg_train = reg_targets[:split_idx_reg]
            y_reg_val   = reg_targets[split_idx_reg:]

        # ── 3. Class weights ─────────────────────────────────────────────
        classes = np.array([0, 1, 2])
        class_weights = np.ones(3)
        sample_weights_train = np.ones(len(y_train))
        if len(y_train) > 0:
            class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
            sample_weights_train = compute_sample_weight("balanced", y=y_train)
            logger.info(
                f"Class weights for {symbol} (SELL/HOLD/BUY): "
                + "/".join(f"{w:.2f}" for w in class_weights)
            )

        # ── 4. Hyperparameter tuning ──────────────────────────────────────
        best_xgb_params: dict = {}
        best_n_estimators = 0
        xgb_val_acc = 0.0

        best_lstm_params: dict = {}
        best_lstm_epochs = 0
        lstm_val_acc = 0.0

        best_ed_params: dict = {}
        best_ed_epochs = 0
        ed_val_acc = 0.0

        prophet_val_acc = 0.0
        best_prophet_cps = 0.1
        best_prophet_sps = 10.0

        if use_xgboost and len(X_tab_train_s) > 0:
            logger.info(f"Tuning XGBoost for {symbol}...")
            best_xgb_params, best_n_estimators, xgb_val_acc = self._tune_xgboost(
                X_tab_train_s, y_train, X_tab_val_s, y_val,
                sample_weight=sample_weights_train,
            )

        if use_lstm and len(X_seq_train_s) > 0:
            logger.info(f"Tuning LSTM for {symbol}...")
            best_lstm_params, best_lstm_epochs, lstm_val_acc = self._tune_lstm(
                X_seq_train_s, y_train, X_seq_val_s, y_val, n_features,
                class_weights=class_weights,
            )

        if use_encoder_decoder and X_seq_reg_train_s is not None:
            logger.info(f"Tuning EncoderDecoder for {symbol}...")
            best_ed_params, best_ed_epochs, ed_val_acc = self._tune_encoder_decoder(
                X_seq_reg_train_s, y_reg_train,
                X_seq_reg_val_s, y_reg_val,
                labels_reg[split_idx_reg:] if labels_reg is not None else None,
                n_features,
            )

        if use_prophet and dates_all is not None:
            logger.info(f"Tuning Prophet for {symbol}...")
            train_end_idx = split_idx + (seq_len if sequences is not None else 0)
            prophet_tuner = ProphetPredictor(horizon=horizon)
            best_prophet_cps, best_prophet_sps, prophet_val_acc = prophet_tuner.tune(
                dates_all, close_all, train_end_idx, seq_len,
                feature_df=prophet_feature_df,
            )

        # ── 5. Derive ensemble weights ────────────────────────────────────
        accs: dict[str, float] = {}
        if use_lstm:          accs["lstm"]           = lstm_val_acc
        if use_xgboost:       accs["xgboost"]        = xgb_val_acc
        if use_encoder_decoder: accs["encoder_decoder"] = ed_val_acc
        if use_prophet:       accs["prophet"]        = prophet_val_acc

        total_acc = sum(accs.values())
        if len(accs) == 1:
            for k in accs: accs[k] = 1.0
        elif total_acc > 0:
            for k in accs: accs[k] = accs[k] / total_acc
        else:
            for k in accs: accs[k] = 1.0 / len(accs)

        lstm_weight   = accs.get("lstm", 0.0)
        xgb_weight    = accs.get("xgboost", 0.0)
        ed_weight     = accs.get("encoder_decoder", 0.0)
        prophet_weight = accs.get("prophet", 0.0)

        logger.info(
            f"Ensemble weights for {symbol}: "
            f"lstm={lstm_weight:.3f}(acc={lstm_val_acc:.4f}), "
            f"xgb={xgb_weight:.3f}(acc={xgb_val_acc:.4f}), "
            f"ed={ed_weight:.3f}(acc={ed_val_acc:.4f}), "
            f"prophet={prophet_weight:.3f}(acc={prophet_val_acc:.4f})"
        )

        # Overall val accuracy = weighted sum of individual accs
        val_accuracy = (
            lstm_weight * lstm_val_acc
            + xgb_weight * xgb_val_acc
            + ed_weight * ed_val_acc
            + prophet_weight * prophet_val_acc
        )

        # ── 6. Retrain final models on full data ───────────────────────────
        logger.info(f"Retraining on full dataset for {symbol}...")

        full_scaler = StandardScaler()
        full_seq_scaler = StandardScaler()
        if tabular is not None and sequences is not None:
            X_tab_full_s = full_scaler.fit_transform(tabular)
            X_seq_full_s = full_seq_scaler.fit_transform(
                sequences.reshape(-1, n_features)
            ).reshape(n_samples, seq_len, n_features)
        else:
            X_tab_full_s = np.zeros((0, n_features))
            X_seq_full_s = np.zeros((0, seq_len if seq_len else 60, n_features))

        class_weights_full = (
            compute_class_weight("balanced", classes=classes, y=labels)
            if labels is not None and len(labels) > 0
            else np.ones(3)
        )
        sample_weights_full = (
            compute_sample_weight("balanced", y=labels)
            if labels is not None and len(labels) > 0
            else np.ones(len(X_tab_full_s))
        )

        xgb_final: XGBoostPredictor | None = None
        lstm_final: LSTMPredictor | None = None
        ed_final: EncoderDecoderPredictor | None = None
        prophet_final: ProphetPredictor | None = None

        if use_xgboost and len(X_tab_full_s) > 0:
            n_est_full = max(best_n_estimators, int(best_n_estimators / self.train_split))
            xgb_final = XGBoostPredictor(
                **best_xgb_params, n_estimators=n_est_full, early_stopping_rounds=None
            )
            xgb_final.train(X_tab_full_s, labels, feature_names=feature_names,
                            sample_weight=sample_weights_full)

        if use_lstm and len(X_seq_full_s) > 0:
            epochs_full = max(best_lstm_epochs, int(best_lstm_epochs / self.train_split))
            lstm_final = LSTMPredictor(input_size=n_features, **best_lstm_params,
                                       epochs=epochs_full)
            lstm_final.train(X_seq_full_s, labels, class_weights=class_weights_full)

        if use_encoder_decoder and seq_reg is not None and reg_targets is not None:
            X_seq_reg_full_s = full_seq_scaler.transform(
                seq_reg.reshape(-1, n_features)
            ).reshape(n_reg, seq_len, n_features)
            epochs_ed_full = max(best_ed_epochs, int(best_ed_epochs / self.train_split))
            ed_final = EncoderDecoderPredictor(
                input_size=n_features, **best_ed_params, epochs=epochs_ed_full,
                horizon=horizon,
            )
            ed_final.train(X_seq_reg_full_s, reg_targets,
                           teacher_forcing_ratio=0.3)   # reduced TF for final model

        if use_prophet and dates_all is not None:
            prophet_final = ProphetPredictor(horizon=horizon)
            prophet_final._changepoint_prior_scale = best_prophet_cps
            prophet_final._seasonality_prior_scale = best_prophet_sps
            prophet_final._residual_std = prophet_tuner._residual_std
            prophet_final._regressors   = prophet_tuner._regressors
            prophet_final.fit_full(dates_all, close_all, feature_df=prophet_feature_df)

        ensemble = EnsembleModel(
            lstm=lstm_final,
            xgboost=xgb_final,
            encoder_decoder=ed_final,
            prophet=prophet_final,
            lstm_weight=lstm_weight,
            xgboost_weight=xgb_weight,
            encoder_decoder_weight=ed_weight,
            prophet_weight=prophet_weight,
        )

        # ── 7. Save ───────────────────────────────────────────────────────
        self._save_models(
            symbol, lstm_final, xgb_final, ed_final, prophet_final,
            full_scaler, full_seq_scaler, feature_names,
            lstm_weight, xgb_weight, ed_weight, prophet_weight,
            selected_models, n_features,
        )

        # ── 8. Generate time-series plots ────────────────────────────────
        plot_paths: dict[str, str] = {}
        try:
            ed_ratios_for_plot = None
            if ed_final is not None and seq_reg is not None:
                X_seq_reg_full_s = full_seq_scaler.transform(
                    seq_reg.reshape(-1, n_features)
                ).reshape(n_reg, seq_len, n_features)
                ed_ratios_for_plot = ed_final.predict_ratios(X_seq_reg_full_s)

            prophet_yhat_hist = (
                prophet_final._historical_yhat if prophet_final is not None else None
            )
            prophet_pred_closes = (
                prophet_final._future_pred_closes if prophet_final is not None else None
            )
            prophet_future_dates = (
                prophet_final._future_pred_dates if prophet_final is not None else None
            )

            if dates_all is not None and close_all is not None:
                plot_save_dir = Path(get_setting("models", "save_dir", default="data/models")).parent / "plots"
                train_end_for_plot = (
                    split_idx + (seq_len if sequences is not None else 0)
                )
                raw_paths = generate_plots(
                    symbol=symbol,
                    dates=dates_all,
                    close=close_all,
                    train_end_idx=train_end_for_plot,
                    horizon=horizon,
                    seq_len=seq_len if sequences is not None else 60,
                    ed_pred_ratios=ed_ratios_for_plot,
                    prophet_yhat_hist=prophet_yhat_hist,
                    prophet_pred_closes=prophet_pred_closes,
                    prophet_future_dates=prophet_future_dates,
                    actual_signals=labels,
                    save_dir=plot_save_dir,
                )
                plot_paths = {k: str(v) for k, v in raw_paths.items()}
                logger.info(f"Plots saved for {symbol}: {list(plot_paths.keys())}")
        except Exception as e:
            logger.warning(f"Plot generation failed for {symbol}: {e}")

        return ensemble, val_accuracy, plot_paths

    # =========================================================================
    # Hyperparameter tuning helpers
    # =========================================================================

    def _tune_xgboost(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[dict, int, float]:
        best_acc, best_params, best_n_est = -1.0, {}, 500
        for params in _XGB_PARAM_GRID:
            model = XGBoostPredictor(**params)
            model.train(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
            acc = balanced_accuracy_score(y_val, model.predict(X_val))
            n_est = getattr(model.model, "best_iteration",
                            self._cfg_xgb_n_estimators()) + 1
            logger.info(f"  XGB {params} → balanced_acc={acc:.4f}, best_iter={n_est}")
            if acc > best_acc:
                best_acc, best_params, best_n_est = acc, dict(params), n_est
        logger.info(f"Best XGB: {best_params}, n_estimators={best_n_est}, balanced_acc={best_acc:.4f}")
        return best_params, best_n_est, best_acc

    def _tune_lstm(
        self,
        X_seq_train: np.ndarray, y_train: np.ndarray,
        X_seq_val: np.ndarray,   y_val: np.ndarray,
        n_features: int,
        class_weights: np.ndarray | None = None,
    ) -> tuple[dict, int, float]:
        best_acc, best_params = -1.0, {}
        best_epochs = get_setting("models", "lstm", "epochs", default=50)
        for params in _LSTM_PARAM_GRID:
            model = LSTMPredictor(input_size=n_features, **params)
            history = model.train(X_seq_train, y_train, X_seq_val, y_val,
                                  class_weights=class_weights)
            acc = balanced_accuracy_score(y_val, model.predict(X_seq_val))
            epoch = int(history.get("best_epoch", model.epochs))
            logger.info(f"  LSTM {params} → balanced_acc={acc:.4f}, best_epoch={epoch}")
            if acc > best_acc:
                best_acc, best_params, best_epochs = acc, dict(params), epoch
        logger.info(f"Best LSTM: {best_params}, epochs={best_epochs}, balanced_acc={best_acc:.4f}")
        return best_params, best_epochs, best_acc

    def _tune_encoder_decoder(
        self,
        X_seq_train: np.ndarray,    # (N_train, seq_len, n_features) — scaled
        y_reg_train: np.ndarray,    # (N_train, horizon) — ratio targets
        X_seq_val: np.ndarray,
        y_reg_val: np.ndarray,
        val_labels: np.ndarray | None,  # (N_val,) classification labels for balanced_acc
        n_features: int,
    ) -> tuple[dict, int, float]:
        """Grid-search ED hyperparameters by MAPE; report balanced accuracy."""
        horizon = int(get_setting("features", "prediction_horizon", default=1))
        best_mape = float("inf")
        best_params: dict = {}
        best_epochs = int(get_setting("models", "encoder_decoder", "epochs", default=50))
        best_val_acc = 0.0

        for params in _ED_PARAM_GRID:
            model = EncoderDecoderPredictor(input_size=n_features, **params, horizon=horizon)
            history = model.train(X_seq_train, y_reg_train, X_seq_val, y_reg_val)
            mape = model.compute_mape(X_seq_val, y_reg_val)
            epoch = int(history.get("best_epoch", model.epochs))

            # Balanced accuracy (binned predictions vs true labels)
            if val_labels is not None and len(val_labels) > 0:
                pred_labels = model.predict(X_seq_val)
                n = min(len(pred_labels), len(val_labels))
                acc = balanced_accuracy_score(val_labels[:n], pred_labels[:n])
            else:
                acc = 0.0

            logger.info(
                f"  ED {params} → MAPE={mape:.4f}, balanced_acc={acc:.4f}, epoch={epoch}"
            )
            if mape < best_mape:
                best_mape = mape
                best_params = dict(params)
                best_epochs = epoch
                best_val_acc = acc

        logger.info(
            f"Best ED: {best_params}, epochs={best_epochs}, "
            f"MAPE={best_mape:.4f}, balanced_acc={best_val_acc:.4f}"
        )
        return best_params, best_epochs, best_val_acc

    def _cfg_xgb_n_estimators(self) -> int:
        return int(get_setting("models", "xgboost", "n_estimators", default=500))

    # =========================================================================
    # Batch training
    # =========================================================================

    def train_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        selected_models: list[str] | None = None,
    ) -> dict[str, dict]:
        """Train models for multiple stocks.

        Returns a dict mapping symbol → result dict with keys:
          status   : 'success' | 'no_data' | 'failed'
          reason   : human-readable failure reason (empty on success)
          accuracy : validation balanced accuracy or None
          model    : EnsembleModel or None
        """
        results: dict[str, dict] = {}
        for i, symbol in enumerate(symbols):
            logger.info(f"Training {i + 1}/{len(symbols)}: {symbol}")
            try:
                model, accuracy, plot_paths = self.train_stock(
                    symbol, start_date, end_date, selected_models
                )
                results[symbol] = {
                    "status": "success",
                    "reason": "",
                    "accuracy": accuracy,
                    "model": model,
                    "plot_paths": plot_paths,
                }
            except ValueError as e:
                logger.error(f"No usable data for {symbol}: {e}")
                results[symbol] = {
                    "status": "no_data",
                    "reason": str(e),
                    "accuracy": None,
                    "model": None,
                    "plot_paths": {},
                }
            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                results[symbol] = {
                    "status": "failed",
                    "reason": str(e),
                    "accuracy": None,
                    "model": None,
                    "plot_paths": {},
                }
        return results

    # =========================================================================
    # Model persistence
    # =========================================================================

    def load_models(
        self, symbol: str
    ) -> tuple[EnsembleModel, StandardScaler, StandardScaler, int | None]:
        """Load trained models for a symbol.

        Returns (ensemble, scaler, seq_scaler, model_age_days).
        Handles three legacy formats in ``selected_models`` for backward compat.
        """
        model_dir = self.save_dir / symbol.replace(".", "_")
        meta_path = model_dir / "meta.joblib"

        if not meta_path.exists():
            raise FileNotFoundError(f"Models not found for {symbol} in {model_dir}")

        meta = joblib.load(meta_path)

        # Resolve selected_models — handle legacy string ``model_mode``
        selected_models: list[str] = meta.get("selected_models", [])
        if not selected_models:
            old_mode = meta.get("model_mode", "ensemble")
            if old_mode == "ensemble":
                selected_models = ["lstm", "xgboost"]
            else:
                selected_models = [old_mode]

        lstm_path = model_dir / "lstm.pt"
        xgb_path  = model_dir / "xgboost.joblib"
        ed_path   = model_dir / "encoder_decoder.pt"
        proph_path = model_dir / "prophet.joblib"

        lstm: LSTMPredictor | None = None
        xgb:  XGBoostPredictor | None = None
        ed:   EncoderDecoderPredictor | None = None
        prophet: ProphetPredictor | None = None

        if "lstm" in selected_models:
            if not lstm_path.exists():
                raise FileNotFoundError(f"LSTM model not found for {symbol}")
            lstm = LSTMPredictor(input_size=meta["input_size"])
            lstm.load(lstm_path)

        if "xgboost" in selected_models:
            if not xgb_path.exists():
                raise FileNotFoundError(f"XGBoost model not found for {symbol}")
            xgb = XGBoostPredictor()
            xgb.load(xgb_path)

        if "encoder_decoder" in selected_models:
            if not ed_path.exists():
                raise FileNotFoundError(f"EncoderDecoder model not found for {symbol}")
            ed = EncoderDecoderPredictor(input_size=meta["input_size"])
            ed.load(ed_path)

        if "prophet" in selected_models:
            if not proph_path.exists():
                raise FileNotFoundError(f"Prophet model not found for {symbol}")
            prophet = ProphetPredictor()
            prophet.load(proph_path)

        ensemble = EnsembleModel(
            lstm=lstm, xgboost=xgb, encoder_decoder=ed, prophet=prophet,
            lstm_weight=meta.get("lstm_weight"),
            xgboost_weight=meta.get("xgb_weight"),
            encoder_decoder_weight=meta.get("ed_weight", 0.0),
            prophet_weight=meta.get("prophet_weight", 0.0),
        )

        model_age_days = None
        trained_at = meta.get("trained_at")
        if trained_at:
            trained_dt = datetime.fromisoformat(trained_at)
            model_age_days = (datetime.now() - trained_dt).days

        return ensemble, meta["scaler"], meta["seq_scaler"], model_age_days

    def _save_models(
        self,
        symbol: str,
        lstm: LSTMPredictor | None,
        xgb: XGBoostPredictor | None,
        ed: EncoderDecoderPredictor | None,
        prophet: ProphetPredictor | None,
        scaler: StandardScaler,
        seq_scaler: StandardScaler,
        feature_names: list[str],
        lstm_weight: float,
        xgb_weight: float,
        ed_weight: float,
        prophet_weight: float,
        selected_models: list[str],
        input_size: int,
    ) -> None:
        model_dir = self.save_dir / symbol.replace(".", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        if lstm    is not None: lstm.save(model_dir / "lstm.pt")
        if xgb     is not None: xgb.save(model_dir / "xgboost.joblib")
        if ed      is not None: ed.save(model_dir / "encoder_decoder.pt")
        if prophet is not None: prophet.save(model_dir / "prophet.joblib")

        joblib.dump(
            {
                "scaler":          scaler,
                "seq_scaler":      seq_scaler,
                "feature_names":   feature_names,
                "input_size":      input_size,
                "lstm_weight":     lstm_weight,
                "xgb_weight":      xgb_weight,
                "ed_weight":       ed_weight,
                "prophet_weight":  prophet_weight,
                "selected_models": selected_models,
                "trained_at":      datetime.now().isoformat(),
            },
            model_dir / "meta.joblib",
        )
        logger.info(f"Saved models for {symbol} to {model_dir}")
