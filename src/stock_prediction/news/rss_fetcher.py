"""Google News RSS fetcher for Indian financial news."""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("news.rss")


@dataclass
class NewsArticle:
    """A single news article."""

    title: str
    source: str
    published: datetime
    url: str
    snippet: str = ""
    query: str = ""
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            self.article_id = hashlib.md5(self.url.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["published"] = self.published.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NewsArticle":
        d["published"] = datetime.fromisoformat(d["published"])
        return cls(**d)


class GoogleNewsRSSFetcher:
    """Fetches Indian financial news from Google News RSS."""

    BASE_URL = "https://news.google.com/rss/search"

    def __init__(self):
        self.cache_dir = Path(get_setting("news", "cache_dir", default="data/news_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = get_setting("news", "cache_expiry_hours", default=6)

    def fetch_market_news(self, max_articles: int | None = None) -> list[NewsArticle]:
        """Fetch general Indian market news."""
        queries = get_setting("news", "queries", default=["Indian stock market"])
        if max_articles is None:
            max_articles = get_setting("news", "max_articles_per_query", default=50)

        all_articles: list[NewsArticle] = []
        seen_ids: set[str] = set()

        for query in queries:
            articles = self._fetch_rss(query, max_articles)
            for article in articles:
                if article.article_id not in seen_ids:
                    seen_ids.add(article.article_id)
                    all_articles.append(article)

        all_articles.sort(key=lambda a: a.published, reverse=True)
        logger.info(f"Fetched {len(all_articles)} unique market news articles")
        return all_articles

    def fetch_stock_news(
        self, stock_name: str, max_articles: int = 20
    ) -> list[NewsArticle]:
        """Fetch news for a specific stock."""
        queries = [
            f"{stock_name} stock NSE",
            f"{stock_name} share price",
        ]
        all_articles: list[NewsArticle] = []
        seen_ids: set[str] = set()

        for query in queries:
            articles = self._fetch_rss(query, max_articles)
            for article in articles:
                if article.article_id not in seen_ids:
                    seen_ids.add(article.article_id)
                    all_articles.append(article)

        all_articles.sort(key=lambda a: a.published, reverse=True)
        return all_articles

    def _fetch_rss(self, query: str, max_articles: int) -> list[NewsArticle]:
        """Fetch articles from Google News RSS for a query."""
        # Check cache
        cached = self._load_cache(query)
        if cached is not None:
            return cached[:max_articles]

        encoded_query = quote_plus(query)
        url = f"{self.BASE_URL}?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            feed = feedparser.parse(url)
            articles = []

            for entry in feed.entries[:max_articles]:
                published = datetime.now()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                # Extract source from title (Google News format: "Title - Source")
                title = entry.get("title", "")
                source = ""
                if " - " in title:
                    parts = title.rsplit(" - ", 1)
                    title = parts[0]
                    source = parts[1]

                snippet = ""
                if hasattr(entry, "summary"):
                    soup = BeautifulSoup(entry.summary, "html.parser")
                    snippet = soup.get_text()[:500]

                articles.append(
                    NewsArticle(
                        title=title,
                        source=source,
                        published=published,
                        url=entry.get("link", ""),
                        snippet=snippet,
                        query=query,
                    )
                )

            self._save_cache(query, articles)
            time.sleep(1)  # Rate limiting
            return articles

        except Exception as e:
            logger.error(f"Failed to fetch RSS for '{query}': {e}")
            return []

    def _cache_path(self, query: str) -> Path:
        key = hashlib.md5(query.encode()).hexdigest()
        return self.cache_dir / f"rss_{key}.json"

    def _load_cache(self, query: str) -> list[NewsArticle] | None:
        path = self._cache_path(query)
        if not path.exists():
            return None

        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cache_expiry_hours:
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return [NewsArticle.from_dict(d) for d in data]
        except Exception:
            return None

    def _save_cache(self, query: str, articles: list[NewsArticle]) -> None:
        path = self._cache_path(query)
        try:
            with open(path, "w") as f:
                json.dump([a.to_dict() for a in articles], f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
