"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 1000 + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n) * 5)
    low = close - np.abs(np.random.randn(n) * 5)
    open_ = close + np.random.randn(n) * 3
    volume = np.random.randint(100000, 10000000, n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def sample_news_articles():
    """Generate sample news articles."""
    from datetime import datetime
    return [
        {
            "title": "Reliance Industries reports strong quarterly earnings",
            "snippet": "Reliance Industries posted record quarterly profit driven by retail and telecom segments.",
            "source": "Economic Times",
            "published": datetime(2024, 1, 15).isoformat(),
            "url": "https://example.com/1",
            "article_id": "abc123",
            "query": "test",
        },
        {
            "title": "TCS wins major IT deal worth $500 million",
            "snippet": "Tata Consultancy Services has won a significant outsourcing deal.",
            "source": "Moneycontrol",
            "published": datetime(2024, 1, 14).isoformat(),
            "url": "https://example.com/2",
            "article_id": "def456",
            "query": "test",
        },
        {
            "title": "HDFC Bank merger integration progressing well",
            "snippet": "HDFC Bank continues to integrate former HDFC Ltd operations smoothly.",
            "source": "LiveMint",
            "published": datetime(2024, 1, 13).isoformat(),
            "url": "https://example.com/3",
            "article_id": "ghi789",
            "query": "test",
        },
    ]


@pytest.fixture
def sample_features():
    """Generate sample feature matrix for model testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 30
    seq_len = 60

    sequences = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    tabular = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, 3, n_samples).astype(np.int64)

    return sequences, tabular, labels
