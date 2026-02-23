"""Technical indicators using the ta library."""

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

    # VWAP approximation (intraday is ideal, but daily approximation)
    df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df[
        "Volume"
    ].cumsum()

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

    # Derived features
    df["SMA_20_50_Cross"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    df["Price_SMA20_Ratio"] = df["Close"] / df["SMA_20"]
    df["Price_SMA50_Ratio"] = df["Close"] / df["SMA_50"]
    df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    logger.info(f"Added {len(df.columns) - 5} technical indicators")
    return df
