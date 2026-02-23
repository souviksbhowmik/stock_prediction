"""Aggregate news articles into per-stock daily features with sentiment windows."""

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher, NewsArticle
from stock_prediction.news.sentiment import FinancialSentimentAnalyzer
from stock_prediction.news.ner import StockEntityLinker
from stock_prediction.utils.constants import NEWS_KEYWORD_CATEGORIES, TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

logger = get_logger("news.features")


class NewsFeatureGenerator:
    """Generate per-stock news features with multi-window sentiment aggregation."""

    def __init__(self):
        self.fetcher = GoogleNewsRSSFetcher()
        self.sentiment_analyzer = FinancialSentimentAnalyzer()
        self.entity_linker = StockEntityLinker()

    def generate_features(
        self,
        symbols: list[str],
        reference_date: datetime | None = None,
    ) -> dict[str, dict[str, float]]:
        """Generate news features for given symbols.

        Returns dict mapping symbol -> feature dict.
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Fetch market news + per-stock news
        all_articles = self.fetcher.fetch_market_news()
        for symbol in symbols:
            name = TICKER_TO_NAME.get(symbol, symbol.replace(".NS", ""))
            stock_articles = self.fetcher.fetch_stock_news(name)
            all_articles.extend(stock_articles)

        # Deduplicate
        seen: set[str] = set()
        unique_articles: list[NewsArticle] = []
        for article in all_articles:
            if article.article_id not in seen:
                seen.add(article.article_id)
                unique_articles.append(article)

        # Analyze sentiment for all articles
        texts = [a.title + ". " + a.snippet for a in unique_articles]
        sentiments = self.sentiment_analyzer.analyze_batch(texts) if texts else []

        # Link articles to stocks
        article_dicts = [a.to_dict() for a in unique_articles]
        stock_articles_map = self.entity_linker.link_articles_to_stocks(article_dicts)

        # Build features per symbol
        features: dict[str, dict[str, float]] = {}
        article_id_to_idx = {a.article_id: i for i, a in enumerate(unique_articles)}

        for symbol in symbols:
            symbol_articles = stock_articles_map.get(symbol, [])
            symbol_sentiments = []

            for art_dict in symbol_articles:
                idx = article_id_to_idx.get(art_dict.get("article_id", ""))
                if idx is not None:
                    symbol_sentiments.append({
                        "sentiment": sentiments[idx],
                        "article": unique_articles[idx],
                    })

            features[symbol] = self._compute_windowed_features(
                symbol_sentiments, reference_date
            )

        return features

    def _compute_windowed_features(
        self,
        articles_with_sentiment: list[dict],
        reference_date: datetime,
    ) -> dict[str, float]:
        """Compute features across 1d, 7d, and 30d windows."""
        features: dict[str, float] = {}
        windows = {"1d": 1, "7d": 7, "30d": 30}

        for window_name, days in windows.items():
            cutoff = reference_date - timedelta(days=days)
            window_items = [
                item for item in articles_with_sentiment
                if item["article"].published >= cutoff
            ]

            scores = [
                item["sentiment"].positive_score - item["sentiment"].negative_score
                for item in window_items
            ]

            n = len(scores)
            features[f"sentiment_{window_name}_mean"] = float(np.mean(scores)) if scores else 0.0
            features[f"sentiment_{window_name}_std"] = float(np.std(scores)) if scores else 0.0
            features[f"news_volume_{window_name}"] = float(n)

            if n > 0:
                pos_count = sum(1 for s in scores if s > 0.1)
                neg_count = sum(1 for s in scores if s < -0.1)
                features[f"positive_ratio_{window_name}"] = pos_count / n
                features[f"negative_ratio_{window_name}"] = neg_count / n
            else:
                features[f"positive_ratio_{window_name}"] = 0.0
                features[f"negative_ratio_{window_name}"] = 0.0

            # Trend (slope) for 7d and 30d windows
            if window_name in ("7d", "30d") and len(scores) >= 3:
                x = np.arange(len(scores))
                slope = np.polyfit(x, scores, 1)[0] if len(scores) > 1 else 0.0
                features[f"sentiment_{window_name}_trend"] = float(slope)
            elif window_name in ("7d", "30d"):
                features[f"sentiment_{window_name}_trend"] = 0.0

            # Keyword category counts
            titles = [item["article"].title.lower() for item in window_items]
            for category, keywords in NEWS_KEYWORD_CATEGORIES.items():
                count = sum(
                    1 for title in titles
                    for kw in keywords
                    if kw in title
                )
                features[f"{category}_{window_name}"] = float(count)

        return features

    def get_news_dataframe(
        self, symbols: list[str], reference_date: datetime | None = None
    ) -> pd.DataFrame:
        """Generate news features as a DataFrame indexed by symbol."""
        features = self.generate_features(symbols, reference_date)
        return pd.DataFrame.from_dict(features, orient="index")
