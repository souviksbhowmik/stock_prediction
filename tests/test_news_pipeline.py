"""Tests for news pipeline."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from stock_prediction.news.rss_fetcher import NewsArticle, GoogleNewsRSSFetcher
from stock_prediction.news.ner import StockEntityLinker


class TestNewsArticle:
    def test_creation(self):
        article = NewsArticle(
            title="Test headline",
            source="TestSource",
            published=datetime(2024, 1, 1),
            url="https://example.com/test",
        )
        assert article.title == "Test headline"
        assert article.article_id  # auto-generated

    def test_to_from_dict(self):
        article = NewsArticle(
            title="Test", source="Src", published=datetime(2024, 1, 1),
            url="https://example.com", snippet="Some text",
        )
        d = article.to_dict()
        restored = NewsArticle.from_dict(d)
        assert restored.title == article.title
        assert restored.url == article.url


class TestStockEntityLinker:
    def test_link_reliance(self):
        linker = StockEntityLinker()
        tickers = linker.link_to_stocks("Reliance Industries reports strong earnings growth")
        assert "RELIANCE.NS" in tickers

    def test_link_tcs(self):
        linker = StockEntityLinker()
        tickers = linker.link_to_stocks("TCS wins major IT outsourcing deal")
        assert "TCS.NS" in tickers

    def test_link_multiple(self):
        linker = StockEntityLinker()
        tickers = linker.link_to_stocks(
            "HDFC Bank and Infosys lead NIFTY gains today"
        )
        assert "HDFCBANK.NS" in tickers
        assert "INFY.NS" in tickers

    def test_no_match(self):
        linker = StockEntityLinker()
        tickers = linker.link_to_stocks("Weather forecast for tomorrow looks sunny")
        assert len(tickers) == 0

    def test_link_articles_to_stocks(self, sample_news_articles):
        linker = StockEntityLinker()
        result = linker.link_articles_to_stocks(sample_news_articles)
        assert "RELIANCE.NS" in result
        assert "TCS.NS" in result
        assert "HDFCBANK.NS" in result
