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
    NIFTY_50_TICKERS,
    SECTOR_MAP,
    TICKER_TO_NAME,
)
from stock_prediction.utils.logging import get_logger

logger = get_logger("signals.screener")


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
