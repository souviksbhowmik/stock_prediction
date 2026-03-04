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
        """Fetch the most recent price for a symbol.

        Tries three sources in order of recency:
        1. ``fast_info.last_price`` — near real-time quote (15-min delayed for
           NSE); available during and shortly after market hours.
        2. ``history(period="1d", interval="1m")`` — 1-minute intraday bars;
           last bar is the most recent traded candle.
        3. ``history(period="7d", interval="1d")`` — end-of-day fallback used
           when intraday data is unavailable (holiday / weekend / after hours).
        """
        ticker = yf.Ticker(symbol)
        now = datetime.now()

        # --- Attempt 1: fast_info last_price (near real-time) ---
        try:
            last_price = getattr(ticker.fast_info, "last_price", None)
            if last_price and float(last_price) > 0:
                last_price = float(last_price)
                df = pd.DataFrame(
                    {
                        "Open": [last_price], "High": [last_price],
                        "Low": [last_price], "Close": [last_price], "Volume": [0],
                    },
                    index=pd.DatetimeIndex([now], name="Date"),
                )
                logger.info(f"[fetch_latest] {symbol} fast_info last_price={last_price:.2f}")
                return StockData(symbol=symbol, df=df)
        except Exception as e:
            logger.debug(f"[fetch_latest] fast_info failed for {symbol}: {e}")

        # --- Attempt 2: 1-minute intraday bars ---
        try:
            df = ticker.history(period="1d", interval="1m")
            if not df.empty:
                df.columns = [c.title().replace(" ", "") for c in df.columns]
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].tail(1)
                df.index = pd.DatetimeIndex(df.index)
                df.index.name = "Date"
                logger.info(
                    f"[fetch_latest] {symbol} 1m intraday close={df['Close'].iloc[-1]:.2f}"
                )
                return StockData(symbol=symbol, df=df)
        except Exception as e:
            logger.debug(f"[fetch_latest] 1m intraday failed for {symbol}: {e}")

        # --- Attempt 3: daily close fallback ---
        logger.warning(f"[fetch_latest] {symbol} falling back to end-of-day close")
        start = now - timedelta(days=7)
        data = self.fetch_historical(
            symbol,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
        )
        if not data.is_empty:
            data.df = data.df.tail(1)
        return data
