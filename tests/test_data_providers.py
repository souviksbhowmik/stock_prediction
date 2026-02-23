"""Tests for data providers."""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from stock_prediction.data import get_provider
from stock_prediction.data.base import DataProvider, StockData


def test_get_provider_yfinance():
    provider = get_provider("yfinance")
    assert isinstance(provider, DataProvider)


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown data provider"):
        get_provider("unknown")


def test_stock_data_empty():
    data = StockData(symbol="TEST.NS", df=pd.DataFrame())
    assert data.is_empty
    assert data.date_range == ("", "")


def test_stock_data_with_data(sample_ohlcv_df):
    data = StockData(symbol="TEST.NS", df=sample_ohlcv_df)
    assert not data.is_empty
    start, end = data.date_range
    assert start < end


@patch("stock_prediction.data.yfinance_provider.yf.Ticker")
def test_yfinance_fetch_historical(mock_ticker_cls, sample_ohlcv_df):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = sample_ohlcv_df
    mock_ticker.info = {"longName": "Test Co", "sector": "IT"}
    mock_ticker_cls.return_value = mock_ticker

    provider = get_provider("yfinance")
    data = provider.fetch_historical("TEST.NS", "2023-01-01", "2023-12-31")

    assert not data.is_empty
    assert data.symbol == "TEST.NS"
    assert "Close" in data.df.columns


@patch("stock_prediction.data.yfinance_provider.yf.Ticker")
def test_yfinance_fetch_empty(mock_ticker_cls):
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_ticker

    provider = get_provider("yfinance")
    data = provider.fetch_historical("INVALID.NS")
    assert data.is_empty
