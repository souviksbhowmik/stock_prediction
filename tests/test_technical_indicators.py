"""Tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np

from stock_prediction.features.technical import add_technical_indicators


def test_add_technical_indicators(sample_ohlcv_df):
    df = add_technical_indicators(sample_ohlcv_df)

    # Check indicator columns exist
    expected_cols = [
        "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
        "SMA_20", "SMA_50", "EMA_12", "EMA_26",
        "OBV", "VWAP", "ATR", "Stoch_K", "Stoch_D",
        "SMA_20_50_Cross", "Price_SMA20_Ratio", "Price_SMA50_Ratio",
        "Volume_SMA20", "Volume_Ratio",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_technical_indicators_values(sample_ohlcv_df):
    df = add_technical_indicators(sample_ohlcv_df)
    df_clean = df.dropna()

    # RSI should be between 0 and 100
    assert df_clean["RSI"].between(0, 100).all()

    # Bollinger Band: upper > middle > lower
    assert (df_clean["BB_Upper"] >= df_clean["BB_Middle"]).all()
    assert (df_clean["BB_Middle"] >= df_clean["BB_Lower"]).all()

    # SMA_20_50_Cross should be 0 or 1
    assert df_clean["SMA_20_50_Cross"].isin([0, 1]).all()


def test_insufficient_data():
    """Should handle short DataFrames gracefully."""
    short_df = pd.DataFrame({
        "Open": [100, 101],
        "High": [102, 103],
        "Low": [98, 99],
        "Close": [101, 102],
        "Volume": [1000, 1100],
    }, index=pd.date_range("2024-01-01", periods=2))

    df = add_technical_indicators(short_df)
    # Should return without crashing, with original columns intact
    assert "Open" in df.columns
    assert "Close" in df.columns


def test_original_data_not_modified(sample_ohlcv_df):
    original_cols = set(sample_ohlcv_df.columns)
    add_technical_indicators(sample_ohlcv_df)
    assert set(sample_ohlcv_df.columns) == original_cols
