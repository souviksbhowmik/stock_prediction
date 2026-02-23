"""Tests for LLM-based news analysis."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from stock_prediction.llm.base import LLMProvider
from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer, BROKER_SCORE_KEYS
from stock_prediction.news.rss_fetcher import NewsArticle


class MockLLMProvider(LLMProvider):
    def __init__(self, response: str = ""):
        self._response = response

    def analyze(self, prompt: str) -> str:
        return self._response

    def analyze_batch(self, prompts: list[str]) -> list[str]:
        return [self._response] * len(prompts)

    def is_available(self) -> bool:
        return True


@pytest.fixture
def sample_articles():
    return [
        NewsArticle(
            title="Company X reports record profits",
            source="Test",
            published=datetime.now(),
            url="https://example.com/1",
        ),
        NewsArticle(
            title="Company X expands into new markets",
            source="Test",
            published=datetime.now(),
            url="https://example.com/2",
        ),
    ]


def test_parse_valid_json_response(sample_articles):
    response = '''{
        "earnings_outlook": 8,
        "competitive_position": 7,
        "management_quality": 6,
        "sector_momentum": 7,
        "risk_level": 4,
        "growth_catalyst": 8,
        "valuation_signal": 6,
        "institutional_interest": 7,
        "macro_impact": 5,
        "overall_broker_score": 7,
        "summary": "Strong growth outlook"
    }'''
    provider = MockLLMProvider(response)
    analyzer = BrokerNewsAnalyzer(provider)
    scores = analyzer.analyze_stock("TEST.NS", sample_articles)

    assert scores["earnings_outlook"] == 8.0
    assert scores["overall_broker_score"] == 7.0
    assert scores["_summary"] == "Strong growth outlook"


def test_parse_invalid_response():
    provider = MockLLMProvider("This is not JSON at all")
    analyzer = BrokerNewsAnalyzer(provider)
    scores = analyzer._parse_response("This is not JSON")

    # Should return neutral scores
    for key in BROKER_SCORE_KEYS:
        assert scores[key] == 5.0


def test_neutral_scores_on_empty_articles():
    provider = MockLLMProvider("")
    analyzer = BrokerNewsAnalyzer(provider)
    scores = analyzer.analyze_stock("TEST.NS", [])

    for key in BROKER_SCORE_KEYS:
        assert scores[key] == 5.0


def test_unavailable_llm():
    provider = MockLLMProvider("")
    provider.is_available = lambda: False
    analyzer = BrokerNewsAnalyzer(provider)
    scores = analyzer.analyze_stock("TEST.NS", [
        NewsArticle(title="Test", source="S", published=datetime.now(), url="http://x.com"),
    ])

    for key in BROKER_SCORE_KEYS:
        assert scores[key] == 5.0


def test_scores_clamped():
    response = '{"earnings_outlook": 15, "competitive_position": -3, "management_quality": 5, "sector_momentum": 5, "risk_level": 5, "growth_catalyst": 5, "valuation_signal": 5, "institutional_interest": 5, "macro_impact": 5, "overall_broker_score": 5}'
    provider = MockLLMProvider(response)
    analyzer = BrokerNewsAnalyzer(provider)
    scores = analyzer._parse_response(response)

    assert scores["earnings_outlook"] == 10.0  # clamped
    assert scores["competitive_position"] == 0.0  # clamped
