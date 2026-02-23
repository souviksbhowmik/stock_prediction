"""Tests for signal generation."""

import pytest

from stock_prediction.signals.generator import SignalGenerator, TradingSignal
from stock_prediction.models.ensemble import EnsemblePrediction


@pytest.fixture
def signal_generator():
    return SignalGenerator()


def _make_prediction(signal: str, confidence: float, probs: dict) -> EnsemblePrediction:
    signal_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
    return EnsemblePrediction(
        signal=signal,
        signal_idx=signal_map[signal],
        confidence=confidence,
        probabilities=probs,
        lstm_probs=probs,
        xgboost_probs=probs,
    )


def test_buy_signal(signal_generator):
    pred = _make_prediction("BUY", 0.7, {"SELL": 0.1, "HOLD": 0.2, "BUY": 0.7})
    signal = signal_generator.generate("TEST.NS", pred)
    assert signal.signal == "BUY"
    assert signal.confidence == 0.7


def test_strong_buy_signal(signal_generator):
    pred = _make_prediction("BUY", 0.85, {"SELL": 0.05, "HOLD": 0.1, "BUY": 0.85})
    signal = signal_generator.generate("TEST.NS", pred)
    assert signal.signal == "STRONG BUY"


def test_sell_signal(signal_generator):
    pred = _make_prediction("SELL", 0.75, {"SELL": 0.75, "HOLD": 0.15, "BUY": 0.1})
    signal = signal_generator.generate("TEST.NS", pred)
    assert signal.signal == "SELL"


def test_strong_sell_signal(signal_generator):
    pred = _make_prediction("SELL", 0.9, {"SELL": 0.9, "HOLD": 0.07, "BUY": 0.03})
    signal = signal_generator.generate("TEST.NS", pred)
    assert signal.signal == "STRONG SELL"


def test_low_confidence_forces_hold(signal_generator):
    pred = _make_prediction("BUY", 0.45, {"SELL": 0.2, "HOLD": 0.35, "BUY": 0.45})
    signal = signal_generator.generate("TEST.NS", pred)
    assert signal.signal == "HOLD"


def test_short_selling_candidate(signal_generator):
    pred = _make_prediction("SELL", 0.8, {"SELL": 0.8, "HOLD": 0.15, "BUY": 0.05})
    tech_data = {"RSI": 75, "MACD_Histogram": -0.5, "Price_SMA50_Ratio": 0.95}
    signal = signal_generator.generate("TEST.NS", pred, tech_data)
    assert signal.is_short_candidate
    assert signal.short_score > 0


def test_not_short_when_buy(signal_generator):
    pred = _make_prediction("BUY", 0.8, {"SELL": 0.05, "HOLD": 0.15, "BUY": 0.8})
    signal = signal_generator.generate("TEST.NS", pred)
    assert not signal.is_short_candidate
    assert signal.short_score == 0.0


def test_generate_batch(signal_generator):
    predictions = {
        "A.NS": _make_prediction("BUY", 0.7, {"SELL": 0.1, "HOLD": 0.2, "BUY": 0.7}),
        "B.NS": _make_prediction("SELL", 0.8, {"SELL": 0.8, "HOLD": 0.1, "BUY": 0.1}),
    }
    signals = signal_generator.generate_batch(predictions)
    assert len(signals) == 2
    # Should be sorted by confidence (highest first)
    assert signals[0].confidence >= signals[1].confidence
