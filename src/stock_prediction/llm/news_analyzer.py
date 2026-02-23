"""LLM-based broker-like news analysis."""

import hashlib
import json
import time
from pathlib import Path

from stock_prediction.config import get_setting
from stock_prediction.llm.base import LLMProvider
from stock_prediction.news.rss_fetcher import NewsArticle
from stock_prediction.utils.constants import TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

logger = get_logger("llm.news_analyzer")

BROKER_ANALYSIS_PROMPT = """You are an experienced Indian stock market analyst. Analyze the following recent news articles about {stock_name} ({symbol}) and provide a broker-like assessment.

Recent News Headlines:
{news_text}

Provide your analysis as a JSON object with numerical scores (0-10, where 0 is very bearish/negative and 10 is very bullish/positive) for each factor:

{{
    "earnings_outlook": <0-10>,
    "competitive_position": <0-10>,
    "management_quality": <0-10>,
    "sector_momentum": <0-10>,
    "risk_level": <0-10>,
    "growth_catalyst": <0-10>,
    "valuation_signal": <0-10>,
    "institutional_interest": <0-10>,
    "macro_impact": <0-10>,
    "overall_broker_score": <0-10>,
    "summary": "<1-2 sentence summary of your view>"
}}

IMPORTANT: Return ONLY the JSON object, no other text. If you lack information for a factor, use 5 (neutral).
"""

BROKER_SCORE_KEYS = [
    "earnings_outlook",
    "competitive_position",
    "management_quality",
    "sector_momentum",
    "risk_level",
    "growth_catalyst",
    "valuation_signal",
    "institutional_interest",
    "macro_impact",
    "overall_broker_score",
]


class BrokerNewsAnalyzer:
    """Uses an LLM to generate broker-like analysis scores from news."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.cache_dir = Path(get_setting("llm", "cache_dir", default="data/news_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = get_setting("llm", "cache_expiry_hours", default=24)

    def analyze_stock(
        self, symbol: str, articles: list[NewsArticle]
    ) -> dict[str, float]:
        """Analyze news articles for a stock and return broker scores.

        Returns dict with score keys mapped to float values (0-10).
        """
        # Check cache
        cached = self._load_cache(symbol)
        if cached is not None:
            return cached

        if not articles:
            logger.info(f"No articles for {symbol}, returning neutral scores")
            return self._neutral_scores()

        if not self.llm.is_available():
            logger.warning("LLM not available, returning neutral scores")
            return self._neutral_scores()

        stock_name = TICKER_TO_NAME.get(symbol, symbol.replace(".NS", ""))
        news_text = "\n".join(
            f"- [{a.published.strftime('%Y-%m-%d')}] {a.title} ({a.source})"
            for a in articles[:15]  # Limit to 15 articles to fit context
        )

        prompt = BROKER_ANALYSIS_PROMPT.format(
            stock_name=stock_name,
            symbol=symbol,
            news_text=news_text,
        )

        response = self.llm.analyze(prompt)
        scores = self._parse_response(response)

        self._save_cache(symbol, scores)
        return scores

    def analyze_batch(
        self,
        stock_articles: dict[str, list[NewsArticle]],
    ) -> dict[str, dict[str, float]]:
        """Analyze news for multiple stocks."""
        results: dict[str, dict[str, float]] = {}
        for symbol, articles in stock_articles.items():
            results[symbol] = self.analyze_stock(symbol, articles)
        return results

    def _parse_response(self, response: str) -> dict[str, float]:
        """Parse LLM JSON response into scores dict."""
        if not response:
            return self._neutral_scores()

        try:
            # Try to extract JSON from response
            text = response.strip()
            # Find JSON block
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)

                scores: dict[str, float] = {}
                for key in BROKER_SCORE_KEYS:
                    val = data.get(key, 5)
                    try:
                        scores[key] = float(min(10, max(0, float(val))))
                    except (ValueError, TypeError):
                        scores[key] = 5.0

                # Store summary if present
                scores["_summary"] = str(data.get("summary", ""))[:200]
                return scores

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return self._neutral_scores()

    def _neutral_scores(self) -> dict[str, float]:
        """Return neutral (5.0) scores for all keys."""
        scores = {key: 5.0 for key in BROKER_SCORE_KEYS}
        scores["_summary"] = ""
        return scores

    def _cache_key(self, symbol: str) -> str:
        today = time.strftime("%Y-%m-%d")
        return hashlib.md5(f"{symbol}_{today}".encode()).hexdigest()

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"llm_{self._cache_key(symbol)}.json"

    def _load_cache(self, symbol: str) -> dict[str, float] | None:
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cache_expiry_hours:
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, symbol: str, scores: dict[str, float]) -> None:
        path = self._cache_path(symbol)
        try:
            with open(path, "w") as f:
                json.dump(scores, f)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")
