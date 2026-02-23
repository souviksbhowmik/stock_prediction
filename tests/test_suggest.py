"""Tests for stock suggestion feature."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from stock_prediction.data.base import StockData
from stock_prediction.news.rss_fetcher import NewsArticle
from stock_prediction.signals.screener import (
    StockScreener,
    StockSuggestion,
    SuggestionResult,
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
    """Sample NewsArticle objects mentioning specific stocks."""
    return [
        NewsArticle(
            title="Reliance Industries reports strong quarterly earnings",
            source="Economic Times",
            published=datetime(2024, 1, 15),
            url="https://example.com/1",
            snippet="Reliance Industries posted record profit.",
        ),
        NewsArticle(
            title="TCS wins major IT deal worth $500 million",
            source="Moneycontrol",
            published=datetime(2024, 1, 14),
            url="https://example.com/2",
            snippet="Tata Consultancy Services has won a deal.",
        ),
        NewsArticle(
            title="Infosys raises guidance amid strong demand",
            source="LiveMint",
            published=datetime(2024, 1, 13),
            url="https://example.com/3",
            snippet="Infosys reported better-than-expected numbers.",
        ),
    ]


def test_suggestion_dataclass():
    s = StockSuggestion(
        rank=1, symbol="TCS.NS", name="TCS", price=3500.0,
        return_1w=2.5, return_1m=5.0, rsi=55.0,
        news_mentions=3, score=7.5, reasons=["Strong momentum"],
    )
    assert s.rank == 1
    assert s.symbol == "TCS.NS"
    assert s.score == 7.5


def test_suggestion_result_dataclass():
    r = SuggestionResult(suggestions=[], total_screened=50, news_articles_scanned=30)
    assert r.total_screened == 50
    assert r.news_articles_scanned == 30
    assert r.suggestions == []


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_returns_correct_count(mock_get_provider, mock_fetcher_cls, mock_provider, mock_news_articles):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = mock_news_articles

    screener = StockScreener()
    result = screener.suggest(count=5, use_news=True)

    assert isinstance(result, SuggestionResult)
    assert len(result.suggestions) <= 5
    assert result.total_screened > 0


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_no_news(mock_get_provider, mock_fetcher_cls, mock_provider):
    mock_get_provider.return_value = mock_provider

    screener = StockScreener()
    result = screener.suggest(count=3, use_news=False)

    assert isinstance(result, SuggestionResult)
    assert result.news_articles_scanned == 0
    # News fetcher should not be instantiated
    mock_fetcher_cls.assert_not_called()
    for s in result.suggestions:
        assert s.news_mentions == 0


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_ranks_descending(mock_get_provider, mock_fetcher_cls, mock_provider, mock_news_articles):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = mock_news_articles

    screener = StockScreener()
    result = screener.suggest(count=10, use_news=True)

    scores = [s.score for s in result.suggestions]
    assert scores == sorted(scores, reverse=True)


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_rank_numbers(mock_get_provider, mock_fetcher_cls, mock_provider):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.suggest(count=5, use_news=True)

    for i, s in enumerate(result.suggestions):
        assert s.rank == i + 1


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_news_mentions_counted(mock_get_provider, mock_fetcher_cls, mock_provider, mock_news_articles):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = mock_news_articles

    screener = StockScreener()
    result = screener.suggest(count=50, use_news=True)

    assert result.news_articles_scanned == 3
    # At least one stock should have news mentions (Reliance, TCS, or Infosys)
    mentioned = [s for s in result.suggestions if s.news_mentions > 0]
    assert len(mentioned) > 0


@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_handles_empty_data(mock_get_provider):
    """Suggest should handle stocks with insufficient data gracefully."""
    import pandas as pd
    provider = MagicMock()
    provider.fetch_historical.return_value = StockData(
        symbol="TEST.NS", df=pd.DataFrame()
    )
    mock_get_provider.return_value = provider

    screener = StockScreener()
    result = screener.suggest(count=5, use_news=False)

    assert isinstance(result, SuggestionResult)
    assert result.total_screened == 0
    assert result.suggestions == []


@patch("stock_prediction.signals.screener.GoogleNewsRSSFetcher")
@patch("stock_prediction.signals.screener.get_provider")
def test_suggest_suggestion_fields(mock_get_provider, mock_fetcher_cls, mock_provider):
    mock_get_provider.return_value = mock_provider
    mock_fetcher_cls.return_value.fetch_market_news.return_value = []

    screener = StockScreener()
    result = screener.suggest(count=1, use_news=True)

    if result.suggestions:
        s = result.suggestions[0]
        assert isinstance(s.symbol, str)
        assert isinstance(s.name, str)
        assert isinstance(s.price, float)
        assert isinstance(s.return_1w, float)
        assert isinstance(s.return_1m, float)
        assert isinstance(s.rsi, float)
        assert isinstance(s.news_mentions, int)
        assert isinstance(s.score, float)
        assert isinstance(s.reasons, list)
        assert len(s.reasons) > 0
