"""Stock screening and selection."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_prediction.config import get_setting
from stock_prediction.data import get_provider
from stock_prediction.features.technical import add_technical_indicators
from stock_prediction.news.news_features import NewsFeatureGenerator
from stock_prediction.llm import get_llm_provider
from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer
from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
from stock_prediction.utils.constants import (
    COMPANY_ALIASES,
    NIFTY_50_TICKERS,
    SECTOR_MAP,
    TICKER_TO_NAME,
)
from stock_prediction.utils.logging import get_logger

logger = get_logger("signals.screener")


@dataclass
class StockSuggestion:
    """A single stock suggestion with composite score."""

    rank: int
    symbol: str
    name: str
    price: float
    return_1w: float
    return_1m: float
    rsi: float
    news_mentions: int
    score: float
    reasons: list[str]


@dataclass
class SuggestionResult:
    """Result from the suggest pipeline."""

    suggestions: list[StockSuggestion]
    total_screened: int
    news_articles_scanned: int


@dataclass
class ScreenerResult:
    """Result from stock screening."""

    top_picks: list[dict] = field(default_factory=list)
    sector_leaders: dict[str, list[dict]] = field(default_factory=dict)
    news_alerts: list[dict] = field(default_factory=list)
    full_rankings: list[dict] = field(default_factory=list)


class StockScreener:
    """Multi-layer stock screening for NIFTY 50."""

    def __init__(self):
        self.data_provider = get_provider(get_setting("data", "provider", default="yfinance"))
        self.volume_spike = get_setting("screener", "volume_spike_threshold", default=1.5)
        self.news_spike = get_setting("screener", "news_volume_spike_threshold", default=2.0)
        self.news_lookback = get_setting("screener", "news_lookback_days", default=3)

    def screen(self, symbols: list[str] | None = None) -> ScreenerResult:
        """Run full screening pipeline."""
        if symbols is None:
            symbols = NIFTY_50_TICKERS

        result = ScreenerResult()

        # Pre-screen for top picks
        result.top_picks = self._pre_screen(symbols)

        # Sector momentum
        result.sector_leaders = self._sector_momentum(symbols)

        # News-driven discovery
        result.news_alerts = self._news_discovery()

        # Full rankings
        result.full_rankings = self._rank_all(symbols)

        return result

    def suggest(self, count: int = 10, use_news: bool = True) -> SuggestionResult:
        """Suggest top NIFTY 50 stocks ranked by technical momentum + news mentions."""
        symbols = NIFTY_50_TICKERS
        stock_scores: list[dict] = []

        # Fetch news once and count mentions per ticker
        news_counts: dict[str, int] = {}
        news_articles_scanned = 0
        if use_news:
            try:
                fetcher = GoogleNewsRSSFetcher()
                articles = fetcher.fetch_market_news()
                news_articles_scanned = len(articles)

                # Build reverse lookup: lowercase alias -> ticker
                alias_to_ticker: dict[str, str] = {}
                for alias, ticker in COMPANY_ALIASES.items():
                    alias_to_ticker[alias.lower()] = ticker
                for ticker, name in TICKER_TO_NAME.items():
                    alias_to_ticker[name.lower()] = ticker
                    # Also add the bare ticker (e.g. "RELIANCE")
                    bare = ticker.replace(".NS", "").lower()
                    alias_to_ticker[bare] = ticker

                for article in articles:
                    text = (article.title + " " + article.snippet).lower()
                    matched_tickers: set[str] = set()
                    for alias, ticker in alias_to_ticker.items():
                        if alias in text:
                            matched_tickers.add(ticker)
                    for ticker in matched_tickers:
                        news_counts[ticker] = news_counts.get(ticker, 0) + 1
            except Exception as e:
                logger.warning(f"News fetch failed for suggestions: {e}")

        # Score each stock
        for symbol in symbols:
            try:
                stock_data = self.data_provider.fetch_historical(symbol)
                if stock_data.is_empty or len(stock_data.df) < 60:
                    continue

                df = add_technical_indicators(stock_data.df)
                if df.empty:
                    continue

                latest = df.iloc[-1]
                score = 0.0
                reasons: list[str] = []

                # --- Momentum returns ---
                price = float(latest["Close"])
                ret_1w = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100
                ret_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100

                momentum = float(ret_1w * 0.6 + ret_1m * 0.4)
                # Normalise momentum into 0-3 range (clamp)
                momentum_score = max(0.0, min(3.0, momentum / 3.0 + 1.5))
                score += momentum_score
                if ret_1w > 2:
                    reasons.append(f"Strong 1W return ({ret_1w:+.1f}%)")
                if ret_1m > 5:
                    reasons.append(f"Strong 1M return ({ret_1m:+.1f}%)")

                # --- Volume spike ---
                if "Volume_Ratio" in latest and latest["Volume_Ratio"] > self.volume_spike:
                    score += 2.0
                    reasons.append(f"Volume spike ({latest['Volume_Ratio']:.1f}x)")

                # --- RSI signal ---
                rsi = float(latest.get("RSI", 50))
                if rsi < 30:
                    score += 1.5
                    reasons.append(f"Oversold (RSI={rsi:.0f})")
                elif rsi > 70:
                    score += 1.0
                    reasons.append(f"Overbought (RSI={rsi:.0f})")

                # --- SMA crossover ---
                if "SMA_20_50_Cross" in latest:
                    sma_cross_prev = df["SMA_20_50_Cross"].iloc[-2] if len(df) > 1 else 0
                    if latest["SMA_20_50_Cross"] == 1 and sma_cross_prev == 0:
                        score += 2.0
                        reasons.append("Bullish SMA 20/50 crossover")

                # --- Near 52-week high ---
                high_52w = df["Close"].tail(252).max() if len(df) >= 252 else df["Close"].max()
                if price >= high_52w * 0.95:
                    score += 1.5
                    reasons.append("Near 52-week high")

                # --- News mentions ---
                mentions = news_counts.get(symbol, 0)
                if mentions > 0:
                    news_score = min(2.0, mentions * 0.5)
                    score += news_score
                    reasons.append(f"{mentions} news mention(s)")

                if not reasons:
                    reasons.append("Baseline momentum")

                stock_scores.append({
                    "symbol": symbol,
                    "name": TICKER_TO_NAME.get(symbol, symbol),
                    "price": price,
                    "return_1w": float(ret_1w),
                    "return_1m": float(ret_1m),
                    "rsi": rsi,
                    "news_mentions": mentions,
                    "score": score,
                    "reasons": reasons,
                })
            except Exception as e:
                logger.warning(f"Suggest scoring failed for {symbol}: {e}")

        # Sort and take top N
        stock_scores.sort(key=lambda x: x["score"], reverse=True)
        suggestions = [
            StockSuggestion(
                rank=i + 1,
                symbol=s["symbol"],
                name=s["name"],
                price=s["price"],
                return_1w=s["return_1w"],
                return_1m=s["return_1m"],
                rsi=s["rsi"],
                news_mentions=s["news_mentions"],
                score=s["score"],
                reasons=s["reasons"],
            )
            for i, s in enumerate(stock_scores[:count])
        ]

        return SuggestionResult(
            suggestions=suggestions,
            total_screened=len(stock_scores),
            news_articles_scanned=news_articles_scanned,
        )

    def _pre_screen(self, symbols: list[str]) -> list[dict]:
        """Identify top picks based on technical + news signals."""
        top_picks = []

        for symbol in symbols:
            try:
                stock_data = self.data_provider.fetch_historical(symbol)
                if stock_data.is_empty or len(stock_data.df) < 60:
                    continue

                df = add_technical_indicators(stock_data.df)
                if df.empty:
                    continue

                latest = df.iloc[-1]
                score = 0.0
                reasons = []

                # Volume spike
                if "Volume_Ratio" in latest and latest["Volume_Ratio"] > self.volume_spike:
                    score += 2.0
                    reasons.append(f"Volume spike ({latest['Volume_Ratio']:.1f}x)")

                # Price near 52-week high
                high_52w = df["Close"].tail(252).max() if len(df) >= 252 else df["Close"].max()
                if latest["Close"] >= high_52w * 0.95:
                    score += 1.5
                    reasons.append("Near 52-week high")

                # Price near 52-week low (potential reversal)
                low_52w = df["Close"].tail(252).min() if len(df) >= 252 else df["Close"].min()
                if latest["Close"] <= low_52w * 1.05:
                    score += 1.0
                    reasons.append("Near 52-week low (reversal candidate)")

                # SMA crossover
                if "SMA_20_50_Cross" in latest:
                    sma_cross_prev = df["SMA_20_50_Cross"].iloc[-2] if len(df) > 1 else 0
                    if latest["SMA_20_50_Cross"] == 1 and sma_cross_prev == 0:
                        score += 2.0
                        reasons.append("Bullish SMA 20/50 crossover")

                # RSI extremes
                if "RSI" in latest:
                    if latest["RSI"] < 30:
                        score += 1.5
                        reasons.append(f"Oversold (RSI={latest['RSI']:.0f})")
                    elif latest["RSI"] > 70:
                        score += 1.0
                        reasons.append(f"Overbought (RSI={latest['RSI']:.0f})")

                if score >= 2.0:
                    top_picks.append({
                        "symbol": symbol,
                        "name": TICKER_TO_NAME.get(symbol, symbol),
                        "score": score,
                        "reasons": reasons,
                        "price": float(latest["Close"]),
                        "rsi": float(latest.get("RSI", 0)),
                        "volume_ratio": float(latest.get("Volume_Ratio", 1)),
                    })
            except Exception as e:
                logger.warning(f"Pre-screen failed for {symbol}: {e}")

        top_picks.sort(key=lambda x: x["score"], reverse=True)
        return top_picks[:10]

    def _sector_momentum(self, symbols: list[str]) -> dict[str, list[dict]]:
        """Calculate sector-level momentum."""
        sector_data: dict[str, list[dict]] = {}

        for sector, sector_symbols in SECTOR_MAP.items():
            sector_stocks = []
            for symbol in sector_symbols:
                if symbol not in symbols:
                    continue
                try:
                    stock_data = self.data_provider.fetch_historical(symbol)
                    if stock_data.is_empty or len(stock_data.df) < 20:
                        continue

                    df = stock_data.df
                    ret_1w = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100
                    ret_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100

                    sector_stocks.append({
                        "symbol": symbol,
                        "name": TICKER_TO_NAME.get(symbol, symbol),
                        "return_1w": float(ret_1w),
                        "return_1m": float(ret_1m),
                        "momentum": float(ret_1w * 0.6 + ret_1m * 0.4),
                    })
                except Exception as e:
                    logger.warning(f"Sector analysis failed for {symbol}: {e}")

            if sector_stocks:
                sector_stocks.sort(key=lambda x: x["momentum"], reverse=True)
                sector_data[sector] = sector_stocks

        return sector_data

    def _news_discovery(self) -> list[dict]:
        """Scan news for stocks outside NIFTY 50 that may warrant attention."""
        alerts = []
        try:
            llm_provider = get_llm_provider(get_setting("llm", "provider", default="ollama"))
            if not llm_provider.is_available():
                logger.info("LLM not available for news discovery")
                return alerts

            fetcher = GoogleNewsRSSFetcher()
            articles = fetcher.fetch_market_news()

            if not articles:
                return alerts

            # Ask LLM to identify non-NIFTY50 stocks mentioned
            headlines = "\n".join(f"- {a.title}" for a in articles[:30])
            prompt = (
                "From these Indian market news headlines, identify any specific stocks or companies "
                "that are NOT in the NIFTY 50 index but seem to have significant news. "
                "Return a JSON array of objects with 'company', 'ticker_guess', and 'reason' fields. "
                "If none found, return []. Only return the JSON.\n\n"
                f"Headlines:\n{headlines}"
            )

            response = llm_provider.analyze(prompt)
            if response:
                import json

                try:
                    start = response.find("[")
                    end = response.rfind("]") + 1
                    if start >= 0 and end > start:
                        data = json.loads(response[start:end])
                        for item in data[:5]:
                            alerts.append({
                                "company": item.get("company", "Unknown"),
                                "ticker": item.get("ticker_guess", ""),
                                "reason": item.get("reason", ""),
                                "source": "LLM News Discovery",
                            })
                except (json.JSONDecodeError, KeyError):
                    pass

        except Exception as e:
            logger.warning(f"News discovery failed: {e}")

        return alerts

    def _rank_all(self, symbols: list[str]) -> list[dict]:
        """Simple ranking of all symbols by recent momentum."""
        rankings = []
        for symbol in symbols:
            try:
                stock_data = self.data_provider.fetch_historical(symbol)
                if stock_data.is_empty or len(stock_data.df) < 20:
                    continue

                df = stock_data.df
                ret_1d = (df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100
                ret_1w = (df["Close"].iloc[-1] / df["Close"].iloc[-5] - 1) * 100

                rankings.append({
                    "symbol": symbol,
                    "name": TICKER_TO_NAME.get(symbol, symbol),
                    "price": float(df["Close"].iloc[-1]),
                    "return_1d": float(ret_1d),
                    "return_1w": float(ret_1w),
                })
            except Exception as e:
                logger.warning(f"Ranking failed for {symbol}: {e}")

        rankings.sort(key=lambda x: x["return_1w"], reverse=True)
        return rankings
