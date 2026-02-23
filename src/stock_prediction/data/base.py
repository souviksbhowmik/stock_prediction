"""Abstract data provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StockData:
    """Container for stock price data."""

    symbol: str
    df: pd.DataFrame  # columns: Open, High, Low, Close, Volume (+ Adj Close)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.df.empty

    @property
    def date_range(self) -> tuple[str, str]:
        if self.is_empty:
            return ("", "")
        return (
            str(self.df.index.min().date()),
            str(self.df.index.max().date()),
        )


class DataProvider(ABC):
    """Abstract base class for stock data providers."""

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> StockData:
        """Fetch historical OHLCV data for a single symbol."""
        ...

    @abstractmethod
    def fetch_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, StockData]:
        """Fetch historical data for multiple symbols."""
        ...

    @abstractmethod
    def fetch_latest(self, symbol: str) -> StockData:
        """Fetch the most recent trading day's data."""
        ...
