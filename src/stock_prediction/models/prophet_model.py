"""Facebook Prophet model for stock price regression → signal classification."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import HORIZON_THRESHOLDS
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.prophet")

# Hyperparameter grid for Prophet changepoint flexibility.
_PROPHET_PARAM_GRID = [
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.10, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.30, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.50, "seasonality_prior_scale": 10.0},
]

# Lag-safe exogenous regressors whose future values can be approximated by
# carrying forward the last known value over the prediction horizon.
# Only columns that actually exist in the feature DataFrame will be used.
CANDIDATE_REGRESSORS: list[str] = [
    "RSI",                    # momentum oscillator (0–100)
    "MACD_Histogram",         # trend momentum
    "BB_PB",                  # Bollinger Band %B (price position in band)
    "ATR",                    # volatility measure
    "NIFTY_Return_1d",        # market direction (last known)
    "Relative_Strength_1d",   # stock vs market (last known)
    "Relative_Strength_5d",   # 5-day relative strength (last known)
    "sentiment_compound",     # news sentiment (slow-moving, 6h cache)
]

# Source columns for lagged features.  For each column listed here, the
# pipeline will create a `{col}_lag{horizon}` column by shifting back by
# `horizon` days.  Because col_lag_h[t+k] = col[t+k-h] for k=1..h, the
# future values are EXACTLY known (all within historical data) — no
# carry-forward approximation is needed.
LAGGED_FEATURE_SOURCES: list[str] = [
    "Volume",
    "MACD",
    "MACD_Signal",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
    "Stoch_K",
    "Stoch_D",
    "BB_Width",
    "OBV",
    "VWAP",
    "Price_Momentum_5d",
    "Volume_Ratio",
    "Price_SMA20_Ratio",
    "Price_SMA50_Ratio",
    "SMA_20_50_Cross",
    "RSI_Change_1d",
    "MACD_Hist_Change_1d",
]


class ProphetPredictor:
    """Prophet-based time-series regressor that predicts future close prices.

    Exogenous regressors
    --------------------
    A subset of ``CANDIDATE_REGRESSORS`` that are present in the feature
    DataFrame is added as Prophet regressors.  For the horizon-ahead forecast,
    their future values are approximated by carry-forward of the last known
    value — reasonable for slowly-moving indicators over 1–10 trading days.

    Training approach
    -----------------
    - Fits Prophet on close price history up to the training split, augmented
      with the available lag-safe regressors.
    - Forecasts forward through all dates using actual feature values for
      historical rows and carry-forward for future rows.
    - Balanced accuracy (per-sample binned predictions) is used for ensemble
      weighting.

    Inference approach
    ------------------
    - Full-data retrain; horizon-ahead forecast uses last known regressor values.
    - A single (3,) probability vector is broadcast over all N ensemble samples.
    """

    def __init__(self, horizon: int | None = None):
        self.horizon = horizon if horizon is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )
        self._thresholds: tuple[float, float] = HORIZON_THRESHOLDS.get(
            self.horizon, (0.022, -0.022)
        )
        self._model = None
        self._residual_std: float = 0.02
        self._current_proba: np.ndarray = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        self._changepoint_prior_scale: float = 0.1
        self._seasonality_prior_scale: float = 10.0
        self._regressors: list[str] = []   # active regressor column names
        # Plot-support attributes populated by fit_full()
        self._historical_yhat: np.ndarray | None = None
        self._future_pred_closes: np.ndarray | None = None
        self._future_pred_dates: pd.DatetimeIndex | None = None

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strip_tz(dates: pd.DatetimeIndex | np.ndarray) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(dates)
        return idx.tz_localize(None) if idx.tz is not None else idx

    def _build_df(
        self,
        dates: pd.DatetimeIndex | np.ndarray,
        close: np.ndarray,
        feature_df: pd.DataFrame | None = None,
        regressors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Build a Prophet-compatible DataFrame with ds, y, and optional regressors."""
        df = pd.DataFrame({
            "ds": self._strip_tz(dates),
            "y":  close.astype(float),
        })
        if regressors and feature_df is not None:
            for col in regressors:
                if col in feature_df.columns:
                    vals = feature_df[col].values
                    # Align length in case of a slice mismatch
                    df[col] = vals[: len(df)]
        return df

    def _fit(
        self,
        df_train: pd.DataFrame,
        regressors: list[str],
        changepoint_prior_scale: float = 0.1,
        seasonality_prior_scale: float = 10.0,
    ):
        """Fit Prophet with optional regressors; returns the fitted model."""
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "prophet is not installed. "
                "Run: conda run -n stock_prediction pip install prophet"
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )
            for reg in regressors:
                if reg in df_train.columns:
                    model.add_regressor(reg)
            model.fit(df_train)
        return model

    def _forecast(
        self,
        model,
        dates: pd.DatetimeIndex | np.ndarray,
        feature_df: pd.DataFrame | None = None,
        regressors: list[str] | None = None,
    ) -> pd.DataFrame:
        """Forecast on a set of specific dates, including regressor values."""
        future = pd.DataFrame({"ds": self._strip_tz(dates)})
        if regressors and feature_df is not None:
            for col in regressors:
                if col in feature_df.columns:
                    vals = feature_df[col].values
                    future[col] = vals[: len(future)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.predict(future)

    # ── Training (with validation evaluation) ─────────────────────────────

    def train(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
        train_end_idx: int,
        seq_len: int,
        feature_df: pd.DataFrame | None = None,
        changepoint_prior_scale: float = 0.1,
        seasonality_prior_scale: float = 10.0,
    ) -> dict:
        """Fit Prophet on the training slice and compute validation balanced accuracy.

        Args:
            dates_all      : DatetimeIndex for all post-dropna rows.
            close_all      : Close prices aligned with dates_all.
            train_end_idx  : Exclusive end index for training (= seq_len + n_train).
            seq_len        : Sequence length, for context only (not used internally).
            feature_df     : DataFrame of lag-safe features aligned with dates_all.
                             Only CANDIDATE_REGRESSORS columns present are used.
        """
        n_df = len(close_all)

        # Select whichever candidate columns are actually available
        regressors = (
            [c for c in CANDIDATE_REGRESSORS if feature_df is not None and c in feature_df.columns]
            if feature_df is not None else []
        )

        df_train = self._build_df(
            dates_all[:train_end_idx],
            close_all[:train_end_idx],
            feature_df.iloc[:train_end_idx] if feature_df is not None else None,
            regressors,
        )
        model = self._fit(df_train, regressors, changepoint_prior_scale, seasonality_prior_scale)

        # Forecast on ALL actual dates using actual feature values
        forecast = self._forecast(model, dates_all, feature_df, regressors)

        balanced_acc = self._compute_val_accuracy(
            close_all, forecast["yhat"].values, train_end_idx, n_df
        )

        # Estimate residual std from val horizon-step residuals
        pred_returns, true_returns = [], []
        for i in range(train_end_idx, n_df):
            j = i + self.horizon
            if j >= n_df:
                break
            pred_r = forecast["yhat"].iloc[j] / close_all[i] - 1
            true_r = close_all[j] / close_all[i] - 1
            pred_returns.append(pred_r)
            true_returns.append(true_r)

        if pred_returns:
            residuals = np.array(true_returns) - np.array(pred_returns)
            self._residual_std = max(float(np.std(residuals)), 1e-4)

        self._model = model
        self._regressors = regressors
        self._changepoint_prior_scale = changepoint_prior_scale
        self._seasonality_prior_scale = seasonality_prior_scale

        logger.info(
            f"Prophet train_end={train_end_idx}, cps={changepoint_prior_scale}, "
            f"regressors={regressors}, balanced_acc={balanced_acc:.4f}, "
            f"residual_std={self._residual_std:.4f}"
        )
        return {"balanced_accuracy": balanced_acc}

    def _compute_val_accuracy(
        self,
        close_all: np.ndarray,
        yhat: np.ndarray,
        train_end_idx: int,
        n_df: int,
    ) -> float:
        """Per-sample validation balanced accuracy using Prophet's forecast."""
        buy_thresh, sell_thresh = self._thresholds
        pred_labels, true_labels = [], []

        for i in range(train_end_idx, n_df):
            j = i + self.horizon
            if j >= n_df:
                break
            if close_all[i] <= 0:
                continue
            pred_r = yhat[j] / close_all[i] - 1
            true_r  = close_all[j] / close_all[i] - 1
            pred_labels.append(2 if pred_r >= buy_thresh else (0 if pred_r <= sell_thresh else 1))
            true_labels.append(2 if true_r >= buy_thresh else (0 if true_r <= sell_thresh else 1))

        if len(pred_labels) < 5:
            return 1.0 / 3.0

        return float(balanced_accuracy_score(true_labels, pred_labels))

    # ── Hyperparameter tuning ─────────────────────────────────────────────

    def tune(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
        train_end_idx: int,
        seq_len: int,
        feature_df: pd.DataFrame | None = None,
    ) -> tuple[float, float, float]:
        """Grid-search changepoint_prior_scale; return (best_cps, best_sps, best_acc)."""
        best_acc, best_cps, best_sps = -1.0, 0.1, 10.0

        for params in _PROPHET_PARAM_GRID:
            result = self.train(
                dates_all, close_all, train_end_idx, seq_len,
                feature_df=feature_df,
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
            )
            acc = result["balanced_accuracy"]
            logger.info(f"  Prophet {params} → balanced_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_cps = params["changepoint_prior_scale"]
                best_sps = params["seasonality_prior_scale"]

        logger.info(
            f"Best Prophet: cps={best_cps}, sps={best_sps}, "
            f"regressors={self._regressors}, balanced_acc={best_acc:.4f}"
        )
        return best_cps, best_sps, best_acc

    # ── Full-data retrain ──────────────────────────────────────────────────

    def fit_full(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
        feature_df: pd.DataFrame | None = None,
    ) -> None:
        """Retrain on the full dataset; cache the horizon-ahead probability.

        Two categories of future regressor values:
        - Lag-safe regressors (CANDIDATE_REGRESSORS): carry-forward last known value.
        - Lagged regressors (*_lag{h} columns): use exact historical values.
          col_lag_h[t+k] = col[t+k-h] for k=1..h — all within known data.
          At future step k: use feature_df[source_col].iloc[-(horizon-k+1)].
          Specifically: step k=1 → feature_df[src].iloc[-horizon],
                        step k=h → feature_df[src].iloc[-1]  (last known).
        """
        regressors = self._regressors   # determined during tune()

        df_full = self._build_df(dates_all, close_all, feature_df, regressors)
        model = self._fit(df_full, regressors,
                          self._changepoint_prior_scale, self._seasonality_prior_scale)
        self._model = model

        # Build future dataframe: all historical dates + horizon future business days
        last_ds = self._strip_tz(dates_all)[-1]
        future_dates = pd.bdate_range(start=last_ds, periods=self.horizon + 1)[1:]

        if regressors and feature_df is not None:
            lag_suffix = f"_lag{self.horizon}"

            # Build one row per future step
            future_row_list = []
            for k in range(1, self.horizon + 1):
                row: dict = {"ds": future_dates[k - 1]}
                for col in regressors:
                    if col not in feature_df.columns:
                        continue
                    if col.endswith(lag_suffix):
                        # Exact historical value: col_lag_h[t+k] = src[t+k-h]
                        # index from end of feature_df: -(horizon - k + 1)
                        src_idx = -(self.horizon - k + 1)
                        src_col = col[: -len(lag_suffix)]
                        if src_col in feature_df.columns:
                            row[col] = float(feature_df[src_col].iloc[src_idx])
                        else:
                            row[col] = float(feature_df[col].iloc[-1])
                    else:
                        # Lag-safe: carry-forward last known value
                        row[col] = float(feature_df[col].iloc[-1])
                future_row_list.append(row)

            future_rows = pd.DataFrame(future_row_list)
            hist_rows = df_full[["ds"] + [c for c in regressors if c in df_full.columns]]
            full_future = pd.concat([hist_rows, future_rows], ignore_index=True)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_future = model.make_future_dataframe(periods=self.horizon, freq="B")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(full_future)

        n_hist = len(dates_all)
        pred_close  = float(forecast["yhat"].iloc[-1])
        yhat_lower  = float(forecast["yhat_lower"].iloc[-1])
        yhat_upper  = float(forecast["yhat_upper"].iloc[-1])
        last_close  = float(close_all[-1])

        pred_return = pred_close / last_close - 1
        interval_sigma = ((yhat_upper - yhat_lower) / 3.92) / max(last_close, 1e-8)
        sigma = max(interval_sigma, self._residual_std)

        self._current_proba = self._return_to_proba(pred_return, sigma)

        # Cache for plot generation
        self._historical_yhat = forecast["yhat"].values[:n_hist]
        self._future_pred_closes = forecast["yhat"].values[n_hist:]
        self._future_pred_dates = future_dates

        logger.info(
            f"Prophet full-data forecast: pred_return={pred_return:.4f}, "
            f"sigma={sigma:.4f}, regressors={regressors}, proba={self._current_proba}"
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def _return_to_proba(self, pred_return: float, sigma: float) -> np.ndarray:
        buy_thresh, sell_thresh = self._thresholds
        p_sell = float(norm.cdf(sell_thresh, loc=pred_return, scale=sigma))
        p_buy  = float(1.0 - norm.cdf(buy_thresh, loc=pred_return, scale=sigma))
        p_hold = float(np.clip(1.0 - p_sell - p_buy, 0.0, 1.0))
        probs  = np.array([p_sell, p_hold, p_buy], dtype=np.float32)
        return probs / max(float(probs.sum()), 1e-8)

    def predict_proba_current(self) -> np.ndarray:
        return self._current_proba.copy()

    def predict_proba(self, N: int) -> np.ndarray:
        return np.tile(self._current_proba, (N, 1))

    def predict(self, N: int) -> np.ndarray:
        return np.full(N, int(np.argmax(self._current_proba)), dtype=np.int64)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model":                   self._model,
                "horizon":                 self.horizon,
                "residual_std":            self._residual_std,
                "current_proba":           self._current_proba,
                "changepoint_prior_scale": self._changepoint_prior_scale,
                "seasonality_prior_scale": self._seasonality_prior_scale,
                "regressors":              self._regressors,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self._model                   = data["model"]
        self.horizon                  = data.get("horizon", self.horizon)
        self._residual_std            = data.get("residual_std", self._residual_std)
        self._current_proba           = data.get("current_proba", self._current_proba)
        self._changepoint_prior_scale = data.get("changepoint_prior_scale", 0.1)
        self._seasonality_prior_scale = data.get("seasonality_prior_scale", 10.0)
        self._regressors              = data.get("regressors", [])
        self._thresholds              = HORIZON_THRESHOLDS.get(self.horizon, (0.022, -0.022))
