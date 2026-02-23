"""yfinance data provider implementation."""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from stock_prediction.config import get_setting
from stock_prediction.data.base import DataProvider, StockData
from stock_prediction.utils.logging import get_logger

logger = get_logger("data.yfinance")


class YFinanceProvider(DataProvider):
    """Data provider using yfinance for NSE stocks."""

    def fetch_historical(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> StockData:
        if start_date is None:
            start_date = get_setting("data", "default_start_date", default="2020-01-01")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return StockData(symbol=symbol, df=pd.DataFrame())

        # Standardize column names
        df.columns = [c.title().replace(" ", "") for c in df.columns]
        # Keep standard OHLCV columns
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep_cols]
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = "Date"

        info = {}
        try:
            info = {
                "name": ticker.info.get("longName", symbol),
                "sector": ticker.info.get("sector", "Unknown"),
                "market_cap": ticker.info.get("marketCap"),
            }
        except Exception:
            pass

        return StockData(symbol=symbol, df=df, metadata=info)

    def fetch_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict[str, StockData]:
        results: dict[str, StockData] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_historical(symbol, start_date, end_date, interval)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = StockData(symbol=symbol, df=pd.DataFrame())
        return results

    def fetch_latest(self, symbol: str) -> StockData:
        end = datetime.now()
        start = end - timedelta(days=7)
        data = self.fetch_historical(
            symbol,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
        if not data.is_empty:
            data.df = data.df.tail(1)
        return data
