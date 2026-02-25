"""Facebook Prophet model for stock price regression → signal classification."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import balanced_accuracy_score

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import HORIZON_THRESHOLDS
from stock_prediction.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("models.prophet")

# Hyperparameter grid for Prophet changepoint flexibility
_PROPHET_PARAM_GRID = [
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.10, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.30, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.50, "seasonality_prior_scale": 10.0},
]


class ProphetPredictor:
    """Prophet-based time-series regressor that predicts future close prices.

    Training approach:
    - Fits Prophet on close price history up to the training split.
    - Forecasts forward through validation dates + horizon ahead.
    - For each validation sample, derives the predicted return from the
      forecast and bins it into SELL / HOLD / BUY.
    - Balanced accuracy over the validation set is used for ensemble weighting.

    Inference approach:
    - Fits Prophet on the full close price history.
    - Forecasts ``horizon`` days ahead from the last available date.
    - Converts the predicted return to a probability vector via Gaussian CDF
      using the residual std estimated from validation residuals.
    - Returns a single (3,) probability that is broadcast over all N samples
      in the ensemble (Prophet gives one signal per horizon, not per historical row).
    """

    def __init__(self, horizon: int | None = None):
        self.horizon = horizon if horizon is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )
        self._thresholds: tuple[float, float] = HORIZON_THRESHOLDS.get(
            self.horizon, (0.022, -0.022)
        )
        self._model = None          # fitted Prophet instance
        self._residual_std: float = 0.02
        # Latest single forecast cached after full-data training
        self._current_proba: np.ndarray = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        self._changepoint_prior_scale: float = 0.1
        self._seasonality_prior_scale: float = 10.0

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_prophet_df(dates: pd.DatetimeIndex | np.ndarray,
                          close: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"ds": pd.DatetimeIndex(dates), "y": close.astype(float)})

    def _fit(self, df_train: pd.DataFrame,
             changepoint_prior_scale: float = 0.1,
             seasonality_prior_scale: float = 10.0):
        """Fit a Prophet model on the given training DataFrame."""
        try:
            from prophet import Prophet  # lazy import to surface install errors early
        except ImportError as exc:
            raise ImportError(
                "prophet is not installed. Run: "
                "conda run -n stock_prediction pip install prophet"
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
            model.fit(df_train)
        return model

    def _forecast_on_dates(self, model, dates: pd.DatetimeIndex | np.ndarray) -> pd.DataFrame:
        """Run Prophet prediction on a specific set of dates."""
        future = pd.DataFrame({"ds": pd.DatetimeIndex(dates)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.predict(future)

    # ── Training (with validation evaluation) ─────────────────────────────

    def train(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
        train_end_idx: int,          # exclusive; Prophet is trained on [:train_end_idx]
        seq_len: int,                # number of leading context rows (for index alignment)
        changepoint_prior_scale: float = 0.1,
        seasonality_prior_scale: float = 10.0,
    ) -> dict:
        """Fit Prophet on training slice and compute validation balanced accuracy.

        Args:
            dates_all      : DatetimeIndex for all rows in the post-dropna df.
            close_all      : Close prices for all rows (same length as dates_all).
            train_end_idx  : Row index (in df space) where training ends (exclusive).
                             Equals seq_len + n_train in the feature-sample split.
            seq_len        : Sequence length (used to map feature samples → df rows).
        """
        n_df = len(close_all)

        df_train = self._build_prophet_df(
            dates_all[:train_end_idx], close_all[:train_end_idx]
        )
        model = self._fit(df_train, changepoint_prior_scale, seasonality_prior_scale)

        # Forecast on ALL actual dates (train + val)
        forecast = self._forecast_on_dates(model, dates_all)
        # forecast['yhat'][i] = Prophet's estimate for row i
        # For i < train_end_idx → in-sample fit; i >= train_end_idx → out-of-sample

        balanced_acc = self._compute_val_accuracy(
            close_all, forecast["yhat"].values, train_end_idx, n_df
        )

        # Estimate residual std from val residuals (final-step only)
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
        self._changepoint_prior_scale = changepoint_prior_scale
        self._seasonality_prior_scale = seasonality_prior_scale

        logger.info(
            f"Prophet train_end_idx={train_end_idx}, "
            f"cps={changepoint_prior_scale}, balanced_acc={balanced_acc:.4f}, "
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
            true_r = close_all[j] / close_all[i] - 1

            pred_labels.append(2 if pred_r >= buy_thresh
                                else (0 if pred_r <= sell_thresh else 1))
            true_labels.append(2 if true_r >= buy_thresh
                                else (0 if true_r <= sell_thresh else 1))

        if len(pred_labels) < 5:
            return 1.0 / 3.0

        return float(balanced_accuracy_score(true_labels, pred_labels))

    # ── Tune grid ─────────────────────────────────────────────────────────

    def tune(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
        train_end_idx: int,
        seq_len: int,
    ) -> tuple[float, float, float]:
        """Grid-search Prophet changepoint_prior_scale; return (best_cps, best_sps, best_acc)."""
        best_acc = -1.0
        best_cps = 0.1
        best_sps = 10.0

        for params in _PROPHET_PARAM_GRID:
            result = self.train(
                dates_all, close_all, train_end_idx, seq_len,
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
            )
            acc = result["balanced_accuracy"]
            logger.info(f"  Prophet {params} → balanced_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_cps = params["changepoint_prior_scale"]
                best_sps = params["seasonality_prior_scale"]

        logger.info(f"Best Prophet: cps={best_cps}, sps={best_sps}, balanced_acc={best_acc:.4f}")
        return best_cps, best_sps, best_acc

    # ── Full-data retrain ──────────────────────────────────────────────────

    def fit_full(
        self,
        dates_all: pd.DatetimeIndex | np.ndarray,
        close_all: np.ndarray,
    ) -> None:
        """Retrain on the complete dataset and cache the horizon-ahead forecast."""
        df_full = self._build_prophet_df(dates_all, close_all)
        model = self._fit(df_full, self._changepoint_prior_scale,
                          self._seasonality_prior_scale)
        self._model = model

        # Forecast horizon days beyond the last date
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            future = model.make_future_dataframe(periods=self.horizon, freq="B")
            forecast = model.predict(future)

        # Cached forecast for the horizon-ahead step
        pred_close = float(forecast["yhat"].iloc[-1])
        yhat_lower = float(forecast["yhat_lower"].iloc[-1])
        yhat_upper = float(forecast["yhat_upper"].iloc[-1])
        last_close = float(close_all[-1])

        pred_return = pred_close / last_close - 1
        # Convert prediction interval width → sigma in return space
        sigma = max(((yhat_upper - yhat_lower) / 3.92) / max(last_close, 1e-8),
                    self._residual_std)

        self._current_proba = self._return_to_proba(pred_return, sigma)
        logger.info(
            f"Prophet full-data forecast: pred_return={pred_return:.4f}, "
            f"sigma={sigma:.4f}, proba={self._current_proba}"
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def _return_to_proba(self, pred_return: float, sigma: float) -> np.ndarray:
        """Convert a scalar predicted return → (3,) SELL/HOLD/BUY probabilities."""
        buy_thresh, sell_thresh = self._thresholds
        p_sell = float(norm.cdf(sell_thresh, loc=pred_return, scale=sigma))
        p_buy = float(1.0 - norm.cdf(buy_thresh, loc=pred_return, scale=sigma))
        p_hold = float(np.clip(1.0 - p_sell - p_buy, 0.0, 1.0))
        probs = np.array([p_sell, p_hold, p_buy], dtype=np.float32)
        return probs / max(probs.sum(), 1e-8)

    def predict_proba_current(self) -> np.ndarray:
        """Return the cached horizon-ahead probability vector (3,)."""
        return self._current_proba.copy()

    def predict_proba(self, N: int) -> np.ndarray:
        """Broadcast the single forecast probability to (N, 3)."""
        return np.tile(self._current_proba, (N, 1))

    def predict(self, N: int) -> np.ndarray:
        """Return class labels (0=SELL, 1=HOLD, 2=BUY) broadcast to (N,)."""
        return np.full(N, int(np.argmax(self._current_proba)), dtype=np.int64)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "horizon": self.horizon,
                "residual_std": self._residual_std,
                "current_proba": self._current_proba,
                "changepoint_prior_scale": self._changepoint_prior_scale,
                "seasonality_prior_scale": self._seasonality_prior_scale,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self._model = data["model"]
        self.horizon = data.get("horizon", self.horizon)
        self._residual_std = data.get("residual_std", self._residual_std)
        self._current_proba = data.get("current_proba", self._current_proba)
        self._changepoint_prior_scale = data.get("changepoint_prior_scale", 0.1)
        self._seasonality_prior_scale = data.get("seasonality_prior_scale", 10.0)
        self._thresholds = HORIZON_THRESHOLDS.get(self.horizon, (0.022, -0.022))
