"""Tests for stock shortlist feature."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from stock_prediction.data.base import StockData
from stock_prediction.news.rss_fetcher import NewsArticle
from stock_prediction.signals.screener import (
    ShortlistResult,
    StockScreener,
    StockSuggestion,
)


@pytest.fixture
def mock_provider(sample_ohlcv_df):
    """Mock data provider returning sample OHLCV data."""
    provider = MagicMock()
    provider.fetch_historical.return_value = StockData(
        symbol="TEST.NS", df=sample_ohlcv_df
    )
    return provider


@pytest.fixture
def mock_news_articles():
    """Sample NewsArticle objects mentioning non-NIFTY stocks."""
    return [
        NewsArticle(
            title="Zomato shares surge on strong quarterly results",
            source="Economic Times",
            published=datetime(2024, 1, 15),
            url="https://example.com/1",
            snippet="Zomato reported better-than-expected earnings.",
        ),
        NewsArticle(
            title="Reliance Industries reports strong quarterly earnings",
            source="Moneycontrol",
            published=datetime(2024, 1, 14),
            url="https://example.com/2",
            snippet="Reliance posted record profit.",
        ),
        NewsArticle(
            title="TCS wins major IT deal",
            source="LiveMint",
            published=datetime(2024, 1, 13),
            url="https://example.com/3",
            snippet="Tata Consultancy Services has won a significant deal.",
        ),
    ]


def test_shortlist_result_dataclass():
    r = ShortlistResult(
        buy_candidates=[], short_candidates=[], trending=[],
        total_screened=50, news_articles_scanned=30,
    )
    assert r.total_screened == 50
    assert r.news_articles_scanned == 30
    assert r.buy_candidates == []
    assert r.short_candidates == []
    assert r.trending == []


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_returns_shortlist_result(mock_get_provider, mock_fetcher_cls, mock_provider, mock_news_articles):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = mock_news_articles

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=True, use_llm=False)

    assert isinstance(result, ShortlistResult)
    assert isinstance(result.buy_candidates, list)
    assert isinstance(result.short_candidates, list)
    assert isinstance(result.trending, list)
    assert result.total_screened > 0


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_buy_candidates_count(mock_get_provider, mock_fetcher_cls, mock_provider):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.shortlist(count=5, use_news=True, use_llm=False)

    assert len(result.buy_candidates) <= 5
    # Buy candidates should be ranked by score descending
    scores = [s.score for s in result.buy_candidates]
    assert scores == sorted(scores, reverse=True)


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_buy_candidates_are_suggestions(mock_get_provider, mock_fetcher_cls, mock_provider):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=True, use_llm=False)

    for s in result.buy_candidates:
        assert isinstance(s, StockSuggestion)
        assert isinstance(s.price, float)
        assert isinstance(s.reasons, list)
        assert len(s.reasons) > 0


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_short_candidates_have_bearish_signals(mock_get_provider, mock_fetcher_cls, mock_provider):
    """Short candidates should have negative momentum or high RSI."""
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.shortlist(count=5, use_news=True, use_llm=False)

    for s in result.short_candidates:
        assert isinstance(s, StockSuggestion)
        # Each short candidate should have at least one bearish reason
        bearish_keywords = ["Weak", "Overbought", "Below SMA", "Bearish"]
        has_bearish = any(
            any(kw.lower() in r.lower() for kw in bearish_keywords)
            for r in s.reasons
        )
        assert has_bearish, f"Short candidate {s.symbol} has no bearish reasons: {s.reasons}"


@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_no_news_returns_empty_trending(mock_get_provider, mock_provider):
    """When use_news=False, trending list should be empty."""
    mock_get_provider.return_value = mock_provider

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=False)

    assert result.trending == []
    assert result.news_articles_scanned == 0


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_handles_fetch_failure_gracefully(mock_get_provider, mock_fetcher_cls):
    """Shortlist should handle yfinance failures for unknown tickers."""
    provider = MagicMock()
    provider.fetch_historical.return_value = StockData(
        symbol="UNKNOWN.NS", df=pd.DataFrame()
    )
    mock_get_provider.return_value = provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=True, use_llm=False)

    assert isinstance(result, ShortlistResult)
    assert result.total_screened == 0
    assert result.buy_candidates == []
    assert result.short_candidates == []


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_trending_are_non_nifty(mock_get_provider, mock_fetcher_cls, sample_ohlcv_df):
    """Trending stocks should be non-NIFTY tickers from news."""
    from stock_prediction.utils.constants import NIFTY_50_TICKERS

    provider = MagicMock()
    provider.fetch_historical.return_value = StockData(
        symbol="TEST.NS", df=sample_ohlcv_df
    )
    mock_get_provider.return_value = provider

    # Create a news article mentioning a non-NIFTY company via COMPANY_ALIASES
    # Use a company name that is in COMPANY_ALIASES but NOT in NIFTY_50_TICKERS
    mock_fetcher_cls.return_value.fetch_market_news.return_value = [
        NewsArticle(
            title="Market update on banking sector",
            source="ET",
            published=datetime(2024, 1, 15),
            url="https://example.com/1",
            snippet="Reliance Industries shows strong growth.",
        ),
    ]

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=True, use_llm=False)

    # Trending should only contain non-NIFTY tickers
    nifty_set = set(NIFTY_50_TICKERS)
    for s in result.trending:
        assert s.symbol not in nifty_set, f"{s.symbol} is in NIFTY 50 but appeared in trending"


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_news_articles_scanned(mock_get_provider, mock_fetcher_cls, mock_provider, mock_news_articles):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = mock_news_articles

    screener = StockScreener()
    result = screener.shortlist(count=3, use_news=True, use_llm=False)

    assert result.news_articles_scanned == len(mock_news_articles)


@patch("stock_prediction.signals.screener.get_provider")
def test_shortlist_rank_numbers(mock_get_provider, mock_provider):
    mock_get_provider.return_value = mock_provider

    screener = StockScreener()
    result = screener.shortlist(count=5, use_news=False)

    for i, s in enumerate(result.buy_candidates):
        assert s.rank == i + 1
    for i, s in enumerate(result.short_candidates):
        assert s.rank == i + 1
