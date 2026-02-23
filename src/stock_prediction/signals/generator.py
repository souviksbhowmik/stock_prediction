"""Signal generation from model predictions."""

from dataclasses import dataclass, field

import numpy as np

from stock_prediction.config import get_setting
from stock_prediction.models.ensemble import EnsemblePrediction
from stock_prediction.utils.logging import get_logger

logger = get_logger("signals.generator")


@dataclass
class TradingSignal:
    """Complete trading signal for a stock."""

    symbol: str
    signal: str  # BUY, STRONG BUY, SELL, STRONG SELL, HOLD
    confidence: float
    strength: float  # 0-1 signal strength
    short_score: float = 0.0  # 0-1 for short selling candidates
    is_short_candidate: bool = False
    probabilities: dict[str, float] = field(default_factory=dict)
    weekly_outlook: str = ""
    monthly_outlook: str = ""
    llm_summary: str = ""
    top_headlines: list[str] = field(default_factory=list)
    technical_summary: dict[str, float] = field(default_factory=dict)


class SignalGenerator:
    """Convert ensemble predictions to trading signals."""

    def __init__(self):
        self.confidence_threshold = get_setting(
            "signals", "confidence_threshold", default=0.6
        )
        self.strong_threshold = get_setting("signals", "strong_threshold", default=0.8)
        self.short_confidence = get_setting(
            "signals", "short_confidence_threshold", default=0.7
        )

    def generate(
        self,
        symbol: str,
        prediction: EnsemblePrediction,
        technical_data: dict[str, float] | None = None,
        llm_summary: str = "",
        headlines: list[str] | None = None,
    ) -> TradingSignal:
        """Generate a trading signal from an ensemble prediction."""
        base_signal = prediction.signal
        confidence = prediction.confidence

        # Force HOLD if confidence is too low
        if confidence < self.confidence_threshold:
            signal = "HOLD"
            strength = confidence
        elif base_signal == "BUY":
            if confidence >= self.strong_threshold:
                signal = "STRONG BUY"
            else:
                signal = "BUY"
            strength = confidence
        elif base_signal == "SELL":
            if confidence >= self.strong_threshold:
                signal = "STRONG SELL"
            else:
                signal = "SELL"
            strength = confidence
        else:
            signal = "HOLD"
            strength = confidence

        # Short selling analysis
        short_score = 0.0
        is_short = False
        if base_signal == "SELL" and confidence >= self.short_confidence:
            short_score = self._compute_short_score(prediction, technical_data)
            is_short = short_score >= 0.5

        # Outlook from probabilities
        weekly_outlook = self._outlook_text(prediction.probabilities, "short-term")
        monthly_outlook = self._outlook_text(prediction.probabilities, "medium-term")

        return TradingSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            strength=strength,
            short_score=short_score,
            is_short_candidate=is_short,
            probabilities=prediction.probabilities,
            weekly_outlook=weekly_outlook,
            monthly_outlook=monthly_outlook,
            llm_summary=llm_summary,
            top_headlines=headlines or [],
            technical_summary=technical_data or {},
        )

    def generate_batch(
        self,
        predictions: dict[str, EnsemblePrediction],
        technical_data: dict[str, dict[str, float]] | None = None,
        llm_summaries: dict[str, str] | None = None,
        headlines: dict[str, list[str]] | None = None,
    ) -> list[TradingSignal]:
        """Generate signals for multiple stocks."""
        signals = []
        for symbol, prediction in predictions.items():
            tech = (technical_data or {}).get(symbol)
            summary = (llm_summaries or {}).get(symbol, "")
            hdl = (headlines or {}).get(symbol, [])
            signals.append(self.generate(symbol, prediction, tech, summary, hdl))

        # Sort by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    def _compute_short_score(
        self,
        prediction: EnsemblePrediction,
        technical_data: dict[str, float] | None,
    ) -> float:
        """Compute short selling score (0-1) based on model + technicals."""
        score = 0.0

        # Base from sell probability
        sell_prob = prediction.probabilities.get("SELL", 0)
        score += sell_prob * 0.4

        if technical_data:
            # RSI overbought (>70)
            rsi = technical_data.get("RSI", 50)
            if rsi > 70:
                score += 0.2 * (rsi - 70) / 30

            # MACD bearish
            macd_hist = technical_data.get("MACD_Histogram", 0)
            if macd_hist < 0:
                score += 0.2

            # Price below SMA50
            price_sma50 = technical_data.get("Price_SMA50_Ratio", 1)
            if price_sma50 < 1.0:
                score += 0.2

        return min(1.0, score)

    def _outlook_text(self, probs: dict[str, float], timeframe: str) -> str:
        """Generate outlook text from probabilities."""
        buy_p = probs.get("BUY", 0)
        sell_p = probs.get("SELL", 0)

        if buy_p > 0.6:
            return f"Bullish {timeframe} outlook ({buy_p:.0%} probability)"
        elif sell_p > 0.6:
            return f"Bearish {timeframe} outlook ({sell_p:.0%} probability)"
        elif buy_p > sell_p:
            return f"Slightly bullish {timeframe} ({buy_p:.0%} buy probability)"
        elif sell_p > buy_p:
            return f"Slightly bearish {timeframe} ({sell_p:.0%} sell probability)"
        return f"Neutral {timeframe} outlook"
