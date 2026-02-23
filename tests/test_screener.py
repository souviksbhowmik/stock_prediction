"""Tests for stock screener."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from stock_prediction.signals.screener import StockScreener, ScreenerResult
from stock_prediction.data.base import StockData


@pytest.fixture
def mock_provider(sample_ohlcv_df):
    """Mock data provider that returns sample data."""
    provider = MagicMock()
    provider.fetch_historical.return_value = StockData(
        symbol="TEST.NS", df=sample_ohlcv_df
    )
    return provider


def test_screener_result_structure():
    result = ScreenerResult()
    assert result.top_picks == []
    assert result.sector_leaders == {}
    assert result.news_alerts == []
    assert result.full_rankings == []


@patch("stock_prediction.signals.screener.get_provider")
def test_rank_all(mock_get_provider, mock_provider, sample_ohlcv_df):
    mock_get_provider.return_value = mock_provider
    screener = StockScreener()
    rankings = screener._rank_all(["TEST.NS"])

    assert len(rankings) == 1
    assert rankings[0]["symbol"] == "TEST.NS"
    assert "price" in rankings[0]
    assert "return_1d" in rankings[0]
    assert "return_1w" in rankings[0]


@patch("stock_prediction.signals.screener.get_provider")
def test_pre_screen(mock_get_provider, mock_provider):
    mock_get_provider.return_value = mock_provider
    screener = StockScreener()
    top_picks = screener._pre_screen(["TEST.NS"])

    # Should return a list (may be empty if no criteria met)
    assert isinstance(top_picks, list)


@patch("stock_prediction.signals.screener.get_provider")
def test_sector_momentum(mock_get_provider, mock_provider):
    mock_get_provider.return_value = mock_provider
    screener = StockScreener()
    sectors = screener._sector_momentum(["TCS.NS", "INFY.NS"])

    assert isinstance(sectors, dict)
