"""Technical indicators using the ta library."""

import numpy as np
import pandas as pd
import ta

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("features.technical")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame.

    Expects columns: Open, High, Low, Close, Volume.
    Returns DataFrame with additional indicator columns.
    """
    df = df.copy()

    if len(df) < 30:
        logger.warning("Insufficient data for technical indicators (need >= 30 rows)")
        return df

    # RSI
    rsi_period = get_setting("features", "technical", "rsi_period", default=14)
    df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"], window=rsi_period).rsi()

    # MACD
    macd_fast = get_setting("features", "technical", "macd_fast", default=12)
    macd_slow = get_setting("features", "technical", "macd_slow", default=26)
    macd_signal = get_setting("features", "technical", "macd_signal", default=9)
    macd = ta.trend.MACD(
        close=df["Close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    # Bollinger Bands
    bb_period = get_setting("features", "technical", "bb_period", default=20)
    bb_std = get_setting("features", "technical", "bb_std", default=2)
    bb = ta.volatility.BollingerBands(close=df["Close"], window=bb_period, window_dev=bb_std)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # SMA
    sma_periods = get_setting("features", "technical", "sma_periods", default=[20, 50])
    for period in sma_periods:
        df[f"SMA_{period}"] = ta.trend.SMAIndicator(close=df["Close"], window=period).sma_indicator()

    # EMA
    ema_periods = get_setting("features", "technical", "ema_periods", default=[12, 26])
    for period in ema_periods:
        df[f"EMA_{period}"] = ta.trend.EMAIndicator(close=df["Close"], window=period).ema_indicator()

    # OBV
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["Close"], volume=df["Volume"]
    ).on_balance_volume()

    # VWAP — rolling 20-day typical-price weighted average (not cumulative)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_vol = typical_price * df["Volume"]
    df["VWAP"] = tp_vol.rolling(window=20).sum() / df["Volume"].rolling(window=20).sum()

    # ATR
    atr_period = get_setting("features", "technical", "atr_period", default=14)
    df["ATR"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=atr_period
    ).average_true_range()

    # Stochastic Oscillator
    stoch_period = get_setting("features", "technical", "stoch_period", default=14)
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=stoch_period
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # Derived ratio features
    df["SMA_20_50_Cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    df["Price_SMA20_Ratio"] = df["Close"] / df["SMA_20"]
    df["Price_SMA50_Ratio"] = df["Close"] / df["SMA_50"]
    df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    # ── Lag / rate-of-change features ────────────────────────────────────
    # These tell the model HOW FAST indicators are moving, not just their level.

    # RSI momentum
    df["RSI_Change_1d"] = df["RSI"].diff(1)
    df["RSI_Change_3d"] = df["RSI"].diff(3)

    # MACD histogram momentum
    df["MACD_Hist_Change_1d"] = df["MACD_Histogram"].diff(1)

    # Price momentum (backward-looking % returns as features)
    df["Price_Momentum_3d"]  = df["Close"].pct_change(3)
    df["Price_Momentum_5d"]  = df["Close"].pct_change(5)
    df["Price_Momentum_10d"] = df["Close"].pct_change(10)

    # Volume momentum
    df["Volume_Change_3d"] = df["Volume"].pct_change(3)

    # Stochastic momentum
    df["Stoch_K_Change_1d"] = df["Stoch_K"].diff(1)

    logger.info(f"Added {len(df.columns) - 5} technical indicators")
    return df


def add_lag_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and explicit trend-direction features for XGBoostLag.

    Expects the df to already have base technical indicators applied
    (Open, High, Low, Close, Volume, RSI, EMA_12, EMA_26, MACD_Histogram,
    Volume_Ratio).

    Adds 16 new columns grouped into four categories:

    Lagged close ratios (3)
        Close_Lag1_Ratio / Close_Lag3_Ratio / Close_Lag5_Ratio
        = Close[t] / Close[t-k].  Values > 1 indicate an uptrend over k days.

    Lagged indicator snapshots (4)
        RSI_Lag3 / RSI_Lag5 / MACD_Hist_Lag3 / Volume_Ratio_Lag3
        Actual past values of key indicators (not just their diffs).

    Trend direction (5)
        ADX / ADX_Pos / ADX_Neg        — trend strength + directional movement
        ADX_Cross                       — 1 if +DI > -DI (bullish trend)
        EMA_12_26_Cross                 — 1 if EMA12 > EMA26 (golden cross)

    Trend slope (2)
        Trend_Slope_10d / Trend_Slope_20d
        Normalised linear-regression slope of Close over the last N days.
        Positive = upward slope; negative = downward slope.

    Price position in recent range (2)
        Price_Rank_10d / Price_Rank_20d
        (Close - rolling_min) / (rolling_max - rolling_min) in [0, 1].
        Values near 1 = near recent high; near 0 = near recent low.
    """
    df = df.copy()

    # ── Lagged close ratios ───────────────────────────────────────────────────
    for k in (1, 3, 5):
        shifted = df["Close"].shift(k)
        df[f"Close_Lag{k}_Ratio"] = df["Close"] / shifted.replace(0, float("nan"))

    # ── Lagged indicator snapshots ────────────────────────────────────────────
    if "RSI" in df.columns:
        df["RSI_Lag3"] = df["RSI"].shift(3)
        df["RSI_Lag5"] = df["RSI"].shift(5)
    if "MACD_Histogram" in df.columns:
        df["MACD_Hist_Lag3"] = df["MACD_Histogram"].shift(3)
    if "Volume_Ratio" in df.columns:
        df["Volume_Ratio_Lag3"] = df["Volume_Ratio"].shift(3)

    # ── ADX — trend strength and direction ────────────────────────────────────
    adx_period = get_setting("features", "technical", "adx_period", default=14)
    adx_ind = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=adx_period
    )
    df["ADX"]      = adx_ind.adx()
    df["ADX_Pos"]  = adx_ind.adx_pos()   # +DI: bullish directional movement
    df["ADX_Neg"]  = adx_ind.adx_neg()   # -DI: bearish directional movement
    df["ADX_Cross"] = (df["ADX_Pos"] > df["ADX_Neg"]).astype(int)

    # ── EMA crossover flag ────────────────────────────────────────────────────
    if "EMA_12" in df.columns and "EMA_26" in df.columns:
        df["EMA_12_26_Cross"] = (df["EMA_12"] > df["EMA_26"]).astype(int)

    # ── Normalised linear-regression slope ───────────────────────────────────
    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        """Slope of OLS fit over `window` days, normalised by mean price."""
        x = np.arange(window, dtype=np.float64)

        def _slope(y: np.ndarray) -> float:
            mean = y.mean()
            if mean == 0:
                return 0.0
            return float(np.polyfit(x, y, 1)[0] / mean)

        return series.rolling(window).apply(_slope, raw=True)

    df["Trend_Slope_10d"] = _rolling_slope(df["Close"], 10)
    df["Trend_Slope_20d"] = _rolling_slope(df["Close"], 20)

    # ── Price position within recent range ───────────────────────────────────
    for w in (10, 20):
        lo = df["Close"].rolling(w).min()
        hi = df["Close"].rolling(w).max()
        rng = (hi - lo).replace(0, float("nan"))
        df[f"Price_Rank_{w}d"] = (df["Close"] - lo) / rng

    logger.info(f"Added {16} lag/trend features (xgboost_lag)")
    return df
