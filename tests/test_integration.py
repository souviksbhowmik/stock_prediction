"""Integration tests for end-to-end workflow."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd

from stock_prediction.data.base import StockData
from stock_prediction.features.technical import add_technical_indicators
from stock_prediction.features.news_based import merge_news_features, merge_llm_features
from stock_prediction.models.lstm_model import LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.models.ensemble import EnsembleModel
from stock_prediction.signals.generator import SignalGenerator


def test_end_to_end_single_stock(sample_ohlcv_df):
    """Test the full pipeline from OHLCV to trading signal."""
    # Step 1: Add technical indicators
    df = add_technical_indicators(sample_ohlcv_df)
    assert len(df.columns) > len(sample_ohlcv_df.columns)

    # Step 2: Add mock news features
    news_features = {
        "sentiment_1d_mean": 0.3,
        "sentiment_7d_mean": 0.2,
        "news_volume_1d": 5.0,
    }
    df = merge_news_features(df, news_features)

    # Step 3: Add mock LLM features
    llm_scores = {
        "earnings_outlook": 7.0,
        "overall_broker_score": 6.5,
    }
    df = merge_llm_features(df, llm_scores)

    # Step 4: Create labels
    df["return_1d"] = df["Close"].pct_change(1).shift(-1)
    conditions = [df["return_1d"] <= -0.01, df["return_1d"] >= 0.01]
    df["signal"] = np.select(conditions, [0, 2], default=1)

    # Drop NaN
    df = df.dropna()
    assert len(df) > 60

    # Step 5: Prepare features
    label_cols = ["return_1d", "signal"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    features = df[feature_cols].values.astype(np.float32)
    labels = df["signal"].values.astype(np.int64)

    # Replace inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    seq_len = 60
    sequences = []
    tabular = []
    seq_labels = []
    for i in range(seq_len, len(features)):
        sequences.append(features[i - seq_len:i])
        tabular.append(features[i])
        seq_labels.append(labels[i])

    sequences = np.array(sequences, dtype=np.float32)
    tabular = np.array(tabular, dtype=np.float32)
    seq_labels = np.array(seq_labels, dtype=np.int64)

    # Train/val split
    split = int(len(sequences) * 0.8)

    # Step 6: Train LSTM
    n_features = sequences.shape[2]
    lstm = LSTMPredictor(input_size=n_features)
    lstm.train(sequences[:split], seq_labels[:split], sequences[split:], seq_labels[split:])

    # Step 7: Train XGBoost
    xgb = XGBoostPredictor()
    xgb.train(tabular[:split], seq_labels[:split], tabular[split:], seq_labels[split:])

    # Step 8: Ensemble prediction
    ensemble = EnsembleModel(lstm, xgb)
    pred = ensemble.predict_single(sequences[-1], tabular[-1])

    assert pred.signal in ("BUY", "HOLD", "SELL")
    assert 0 <= pred.confidence <= 1

    # Step 9: Generate trading signal
    gen = SignalGenerator()
    signal = gen.generate("TEST.NS", pred)

    assert signal.symbol == "TEST.NS"
    assert signal.signal in ("BUY", "STRONG BUY", "HOLD", "SELL", "STRONG SELL")


def test_config_loading():
    """Test that config loads without errors."""
    from stock_prediction.config import load_settings, get_setting

    settings = load_settings()
    assert "data" in settings
    assert "models" in settings

    val = get_setting("models", "lstm", "epochs")
    assert isinstance(val, int)
    assert val > 0


def test_constants():
    """Test that constants are properly defined."""
    from stock_prediction.utils.constants import (
        NIFTY_50_TICKERS,
        COMPANY_ALIASES,
        TICKER_TO_NAME,
        SECTOR_MAP,
    )

    assert len(NIFTY_50_TICKERS) == 50
    assert all(t.endswith(".NS") for t in NIFTY_50_TICKERS)
    assert "reliance" in COMPANY_ALIASES
    assert COMPANY_ALIASES["reliance"] == "RELIANCE.NS"
    assert "RELIANCE.NS" in TICKER_TO_NAME
    assert len(SECTOR_MAP) > 0
