"""Data provider factory."""

from stock_prediction.data.base import DataProvider, StockData


def get_provider(name: str = "yfinance") -> DataProvider:
    """Factory to get a data provider by name."""
    if name == "yfinance":
        from stock_prediction.data.yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    raise ValueError(f"Unknown data provider: {name}")


__all__ = ["get_provider", "DataProvider", "StockData"]
