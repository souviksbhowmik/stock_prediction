"""Utilities for generating time-series training/validation/prediction plots.

Three HTML plot files are produced per trained stock:
  <symbol>_train_plot.html — actual vs predicted on the training period only
  <symbol>_val_plot.html   — actual vs predicted for train+val with split marker
  <symbol>_pred_plot.html  — full history + horizon-ahead future forecast

"First-value" rule for Encoder-Decoder multi-step predictions:
  For close-array index d, the oldest (most reliable) prediction comes from
  regression sample j = d - seq_len - horizon, which predicted d as its
  final step.  pred_close[d] = close[d - horizon] * ed_pred_ratios[j, -1]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stock_prediction.utils.logging import get_logger

logger = get_logger("utils.plot_utils")

_SIGNAL_COLOR = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}  # SELL/HOLD/BUY
_SIGNAL_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}


def _strip_tz(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(dates)
    return idx.tz_localize(None) if idx.tz is not None else idx


def generate_plots(
    symbol: str,
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    train_end_idx: int,
    horizon: int,
    seq_len: int = 60,
    ed_pred_ratios: np.ndarray | None = None,
    prophet_yhat_hist: np.ndarray | None = None,
    prophet_pred_closes: np.ndarray | None = None,
    prophet_future_dates: pd.DatetimeIndex | None = None,
    actual_signals: np.ndarray | None = None,
    predicted_signals: np.ndarray | None = None,
    save_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate and save three HTML Plotly plots for a trained stock.

    Parameters
    ----------
    symbol           : Stock ticker (used in file names and plot titles).
    dates            : DatetimeIndex aligned with ``close``.
    close            : Actual close prices, shape (n,).
    train_end_idx    : Index in ``close`` where the training period ends
                       (validation starts); used to draw the split marker.
    horizon          : Prediction horizon in days.
    seq_len          : Sequence length used by LSTM / ED models (default 60).
    ed_pred_ratios   : (n_reg, horizon) ratios from ED full-data inference,
                       or None if Encoder-Decoder was not trained.
    prophet_yhat_hist: Prophet fitted values for all historical dates (len=n).
    prophet_pred_closes: Prophet predicted close prices for the next
                        ``horizon`` business days (shape (horizon,)).
    prophet_future_dates: Business-day DatetimeIndex of length ``horizon``
                         corresponding to ``prophet_pred_closes``.
    actual_signals   : (n_samples,) true signal labels from prepare_training_data,
                       starting at close-array index (n - n_samples).
                       Shown as filled circles (actual outcomes).
    predicted_signals: (n_samples,) signal labels predicted by the trained
                       model (LSTM / XGBoost) on the full dataset.
                       Shown as triangles so they can be compared with the
                       actual outcomes at a glance.
    save_dir         : Directory under which a per-symbol subdirectory is
                       created (default ``data/plots``).

    Returns
    -------
    dict with keys "train_plot", "val_plot", "pred_plot" → Path to .html file.
    """
    try:
        import plotly.graph_objects as go
        import plotly.offline as pyo
    except ImportError:
        logger.error("plotly is not installed — skipping plot generation")
        return {}

    save_dir = save_dir or Path("data/plots")
    sym_dir = save_dir / symbol.replace(".", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)

    dates_arr = _strip_tz(dates)
    n = len(close)

    # ── Reconstruct ED predicted close (first-value rule) ─────────────────
    # "First value" for date d: the oldest prediction is from regression
    # sample j = d - seq_len - horizon, predicting d as the horizon-th step.
    #   pred_close[d] = close[d - horizon] * ed_pred_ratios[j, -1]
    ed_pred_close: np.ndarray | None = None
    if ed_pred_ratios is not None and len(ed_pred_ratios) > 0:
        ed_close = np.full(n, np.nan)
        n_reg = len(ed_pred_ratios)
        for d in range(seq_len + horizon, n):
            j = d - seq_len - horizon
            if 0 <= j < n_reg:
                src = d - horizon
                if 0 <= src < n:
                    ed_close[d] = close[src] * ed_pred_ratios[j, -1]
        ed_pred_close = ed_close

    # ── Align actual_signals to full close-array indices ──────────────────
    def _align_signals(signals: np.ndarray) -> np.ndarray:
        full = np.full(n, -1, dtype=np.int64)
        start = n - len(signals)
        if start >= 0:
            full[start:] = signals
        else:
            full[:] = signals[-n:]
        return full

    sig_arr: np.ndarray | None = None
    if actual_signals is not None and len(actual_signals) > 0:
        sig_arr = _align_signals(actual_signals)

    pred_sig_arr: np.ndarray | None = None
    if predicted_signals is not None and len(predicted_signals) > 0:
        pred_sig_arr = _align_signals(predicted_signals)

    # ── Helper builders ───────────────────────────────────────────────────
    def _line(x, y, name, color, dash=None, width=1.5):
        return go.Scatter(
            x=list(x), y=list(y), mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash),
        )

    def _signal_traces(mask):
        """Actual-outcome signal markers — filled circles."""
        if sig_arr is None:
            return []
        out = []
        for sv, color in _SIGNAL_COLOR.items():
            m = mask & (sig_arr == sv)
            if not m.any():
                continue
            out.append(go.Scatter(
                x=list(dates_arr[m]), y=list(close[m]), mode="markers",
                name=f"Actual {_SIGNAL_LABEL[sv]}",
                marker=dict(color=color, size=5, opacity=0.75, symbol="circle"),
                showlegend=True,
            ))
        return out

    _PRED_SYMBOL = {0: "triangle-down", 1: "diamond", 2: "triangle-up"}

    def _predicted_signal_traces(mask):
        """Model-predicted signal markers — triangles (BUY ▲, SELL ▼)."""
        if pred_sig_arr is None:
            return []
        out = []
        for sv, color in _SIGNAL_COLOR.items():
            m = mask & (pred_sig_arr == sv)
            if not m.any():
                continue
            out.append(go.Scatter(
                x=list(dates_arr[m]), y=list(close[m]), mode="markers",
                name=f"Pred {_SIGNAL_LABEL[sv]}",
                marker=dict(
                    color=color, size=7, opacity=0.55,
                    symbol=_PRED_SYMBOL[sv],
                    line=dict(color="white", width=0.5),
                ),
                showlegend=True,
            ))
        return out

    split_date_str = str(dates_arr[train_end_idx]) if 0 < train_end_idx < n else None

    _layout = dict(
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
        margin=dict(t=80, b=60),
    )

    # ─────────────────────────────────────────────────────────────────────
    # Plot 1 — Training period only
    # ─────────────────────────────────────────────────────────────────────
    end = min(train_end_idx, n) if train_end_idx > 0 else n
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[:end] = True

    fig_train = go.Figure()
    fig_train.add_trace(_line(dates_arr[:end], close[:end], "Actual Close", "#1d4ed8"))

    if ed_pred_close is not None:
        valid = tr_mask & ~np.isnan(ed_pred_close)
        if valid.any():
            fig_train.add_trace(_line(
                dates_arr[valid], ed_pred_close[valid], "ED Predicted", "#f97316", dash="dot"
            ))

    if prophet_yhat_hist is not None:
        ylen = min(end, len(prophet_yhat_hist))
        fig_train.add_trace(_line(
            dates_arr[:ylen], prophet_yhat_hist[:ylen], "Prophet Fit", "#8b5cf6", dash="dot"
        ))

    for t in _signal_traces(tr_mask):
        fig_train.add_trace(t)
    for t in _predicted_signal_traces(tr_mask):
        fig_train.add_trace(t)

    fig_train.update_layout(
        title=f"{symbol} — Training Period (horizon={horizon}d)",
        xaxis_title="Date", yaxis_title="Price (₹)", **_layout,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Plot 2 — Full train + validation with split marker
    # ─────────────────────────────────────────────────────────────────────
    full_mask = np.ones(n, dtype=bool)

    fig_val = go.Figure()
    fig_val.add_trace(_line(dates_arr, close, "Actual Close", "#1d4ed8"))

    if ed_pred_close is not None:
        valid = ~np.isnan(ed_pred_close)
        if valid.any():
            fig_val.add_trace(_line(
                dates_arr[valid], ed_pred_close[valid], "ED Predicted", "#f97316", dash="dot"
            ))

    if prophet_yhat_hist is not None:
        ylen = min(n, len(prophet_yhat_hist))
        fig_val.add_trace(_line(
            dates_arr[:ylen], prophet_yhat_hist[:ylen], "Prophet Fit", "#8b5cf6", dash="dot"
        ))

    for t in _signal_traces(full_mask):
        fig_val.add_trace(t)
    for t in _predicted_signal_traces(full_mask):
        fig_val.add_trace(t)

    if split_date_str:
        fig_val.add_shape(
            type="line",
            x0=split_date_str, x1=split_date_str, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="#6b7280", dash="dash", width=1.5),
        )
        fig_val.add_annotation(
            x=split_date_str, y=1, xref="x", yref="paper",
            text="Train/Val split", showarrow=False,
            yanchor="bottom", font=dict(size=11, color="#6b7280"),
        )

    fig_val.update_layout(
        title=f"{symbol} — Train + Validation Fit (horizon={horizon}d)",
        xaxis_title="Date", yaxis_title="Price (₹)", **_layout,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Plot 3 — Full history + future forecast
    # ─────────────────────────────────────────────────────────────────────
    fig_pred = go.Figure()
    fig_pred.add_trace(_line(dates_arr, close, "Actual Close", "#1d4ed8"))

    # Prophet future forecast
    if prophet_pred_closes is not None and prophet_future_dates is not None:
        fut_d = _strip_tz(prophet_future_dates)
        conn_x = [dates_arr[-1]] + list(fut_d)
        conn_y = [float(close[-1])] + [float(v) for v in prophet_pred_closes]
        fig_pred.add_trace(go.Scatter(
            x=conn_x, y=conn_y, mode="lines+markers",
            name=f"Prophet Forecast (+{horizon}d)",
            line=dict(color="#8b5cf6", width=2, dash="dot"),
            marker=dict(size=7),
        ))

    for t in _signal_traces(np.ones(n, dtype=bool)):
        fig_pred.add_trace(t)
    for t in _predicted_signal_traces(np.ones(n, dtype=bool)):
        fig_pred.add_trace(t)

    # ED future forecast — use last regression sample's ratios
    if ed_pred_ratios is not None and len(ed_pred_ratios) > 0:
        last_close = float(close[-1])
        last_ratios = ed_pred_ratios[-1]   # (horizon,)
        ed_fut_close = last_close * last_ratios
        ed_fut_dates = pd.bdate_range(start=dates_arr[-1], periods=horizon + 1)[1:]
        conn_x = [dates_arr[-1]] + list(ed_fut_dates)
        conn_y = [last_close] + [float(v) for v in ed_fut_close]
        fig_pred.add_trace(go.Scatter(
            x=conn_x, y=conn_y, mode="lines+markers",
            name=f"ED Forecast (+{horizon}d)",
            line=dict(color="#f97316", width=2, dash="dot"),
            marker=dict(size=7),
        ))

    fig_pred.update_layout(
        title=f"{symbol} — Price Forecast (+{horizon} days)",
        xaxis_title="Date", yaxis_title="Price (₹)", **_layout,
    )

    # ── Save HTML files ───────────────────────────────────────────────────
    paths: dict[str, Path] = {}
    for key, fig, fname in [
        ("train_plot", fig_train, f"{symbol}_train_plot.html"),
        ("val_plot",   fig_val,   f"{symbol}_val_plot.html"),
        ("pred_plot",  fig_pred,  f"{symbol}_pred_plot.html"),
    ]:
        out = sym_dir / fname
        pyo.plot(fig, filename=str(out), auto_open=False, include_plotlyjs="cdn")
        paths[key] = out
        logger.info(f"Saved {key} → {out}")

    return paths
