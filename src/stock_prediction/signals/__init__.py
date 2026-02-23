"""Signal generation, screening, and reporting."""

from stock_prediction.signals.generator import SignalGenerator, TradingSignal
from stock_prediction.signals.screener import StockScreener, ScreenerResult
from stock_prediction.signals.report import ReportFormatter
from stock_prediction.signals.paper_trading import PaperTrade, PaperTradingManager, GainReport

__all__ = [
    "SignalGenerator",
    "TradingSignal",
    "StockScreener",
    "ScreenerResult",
    "ReportFormatter",
    "PaperTrade",
    "PaperTradingManager",
    "GainReport",
]
