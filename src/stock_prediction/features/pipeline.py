"""Feature pipeline: orchestrate OHLCV, technical, news, and LLM features."""

from datetime import datetime

import numpy as np
import pandas as pd

from stock_prediction.config import get_setting
from stock_prediction.data import get_provider
from stock_prediction.features.technical import add_technical_indicators
from stock_prediction.features.news_based import merge_news_features, merge_llm_features
from stock_prediction.news.news_features import NewsFeatureGenerator
from stock_prediction.llm import get_llm_provider
from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer
from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
from stock_prediction.utils.constants import TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

logger = get_logger("features.pipeline")


class FeaturePipeline:
    """Build full feature matrix for model training/prediction."""

    def __init__(self, use_news: bool = True, use_llm: bool = True):
        self.data_provider = get_provider(get_setting("data", "provider", default="yfinance"))
        self.use_news = use_news
        self.use_llm = use_llm
        self.sequence_length = get_setting("features", "sequence_length", default=60)

        if use_news:
            self.news_generator = NewsFeatureGenerator()
        if use_llm:
            try:
                llm_provider = get_llm_provider(get_setting("llm", "provider", default="ollama"))
                self.broker_analyzer = BrokerNewsAnalyzer(llm_provider)
                self.news_fetcher = GoogleNewsRSSFetcher()
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Disabling LLM features.")
                self.use_llm = False

    def build_features(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Build complete feature DataFrame for a symbol."""
        # Step 1: Fetch OHLCV
        stock_data = self.data_provider.fetch_historical(symbol, start_date, end_date)
        if stock_data.is_empty:
            logger.error(f"No data for {symbol}")
            return pd.DataFrame()

        df = stock_data.df

        # Step 2: Technical indicators
        df = add_technical_indicators(df)

        # Step 3: News features
        if self.use_news:
            try:
                news_features = self.news_generator.generate_features([symbol])
                symbol_news = news_features.get(symbol, {})
                df = merge_news_features(df, symbol_news)
            except Exception as e:
                logger.warning(f"News features failed for {symbol}: {e}")

        # Step 4: LLM broker features
        if self.use_llm:
            try:
                name = TICKER_TO_NAME.get(symbol, symbol.replace(".NS", ""))
                articles = self.news_fetcher.fetch_stock_news(name)
                llm_scores = self.broker_analyzer.analyze_stock(symbol, articles)
                df = merge_llm_features(df, llm_scores)
            except Exception as e:
                logger.warning(f"LLM features failed for {symbol}: {e}")

        # Step 5: Create labels
        df = self._add_labels(df)

        # Drop rows with NaN (from indicator warmup)
        df = df.dropna()

        logger.info(f"Built features for {symbol}: {df.shape}")
        return df

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return and signal labels."""
        buy_thresh = get_setting("signals", "buy_return_threshold", default=0.01)
        sell_thresh = get_setting("signals", "sell_return_threshold", default=-0.01)

        df = df.copy()
        df["return_1d"] = df["Close"].pct_change(1).shift(-1)
        df["return_5d"] = df["Close"].pct_change(5).shift(-5)

        # Signal: 0=SELL, 1=HOLD, 2=BUY
        conditions = [
            df["return_1d"] <= sell_thresh,
            df["return_1d"] >= buy_thresh,
        ]
        choices = [0, 2]
        df["signal"] = np.select(conditions, choices, default=1)

        return df

    def prepare_training_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare data for model training.

        Returns:
            (sequences, tabular, labels, feature_names)
            - sequences: shape (N, seq_len, n_features) for LSTM
            - tabular: shape (N, n_features) for XGBoost
            - labels: shape (N,) signal labels
            - feature_names: list of feature column names
        """
        df = self.build_features(symbol, start_date, end_date)
        if df.empty or len(df) < self.sequence_length + 10:
            logger.error(f"Insufficient data for {symbol}")
            return np.array([]), np.array([]), np.array([]), []

        # Separate features and labels
        label_cols = ["return_1d", "return_5d", "signal"]
        feature_cols = [c for c in df.columns if c not in label_cols]
        feature_names = feature_cols

        features = df[feature_cols].values
        labels = df["signal"].values

        # Create sequences for LSTM
        sequences = []
        tabular = []
        seq_labels = []

        for i in range(self.sequence_length, len(features)):
            sequences.append(features[i - self.sequence_length : i])
            tabular.append(features[i])
            seq_labels.append(labels[i])

        sequences = np.array(sequences, dtype=np.float32)
        tabular = np.array(tabular, dtype=np.float32)
        seq_labels = np.array(seq_labels, dtype=np.int64)

        # Replace any remaining inf/nan
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        tabular = np.nan_to_num(tabular, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            f"Prepared {symbol}: sequences={sequences.shape}, "
            f"tabular={tabular.shape}, labels={seq_labels.shape}"
        )
        return sequences, tabular, seq_labels, feature_names
