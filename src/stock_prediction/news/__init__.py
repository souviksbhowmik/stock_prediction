"""News pipeline: RSS fetching, sentiment analysis, NER, feature generation."""

from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher, NewsArticle
from stock_prediction.news.sentiment import FinancialSentimentAnalyzer
from stock_prediction.news.ner import StockEntityLinker
from stock_prediction.news.news_features import NewsFeatureGenerator

__all__ = [
    "GoogleNewsRSSFetcher",
    "NewsArticle",
    "FinancialSentimentAnalyzer",
    "StockEntityLinker",
    "NewsFeatureGenerator",
]
