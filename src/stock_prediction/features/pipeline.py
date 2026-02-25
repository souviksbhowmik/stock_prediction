"""Feature pipeline: orchestrate OHLCV, technical, news, and LLM features."""

from datetime import datetime

import numpy as np
import pandas as pd

from stock_prediction.config import get_setting
from stock_prediction.data import get_provider
from stock_prediction.features.technical import add_technical_indicators
from stock_prediction.features.news_based import merge_news_features, merge_llm_features
from stock_prediction.features.financial import FinancialFeatureGenerator
from stock_prediction.news.news_features import NewsFeatureGenerator
from stock_prediction.llm import get_llm_provider
from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer
from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
from stock_prediction.utils.constants import TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

logger = get_logger("features.pipeline")

# Per-horizon buy/sell thresholds derived from √t volatility scaling
# (base ±1% at horizon=1, scaled by √horizon for longer horizons).
# Restricted to horizons 1-10; values are (buy_thresh, sell_thresh).
HORIZON_THRESHOLDS: dict[int, tuple[float, float]] = {
    1:  ( 0.010, -0.010),
    2:  ( 0.014, -0.014),
    3:  ( 0.017, -0.017),
    4:  ( 0.020, -0.020),
    5:  ( 0.022, -0.022),
    6:  ( 0.024, -0.024),
    7:  ( 0.026, -0.026),
    8:  ( 0.028, -0.028),
    9:  ( 0.030, -0.030),
    10: ( 0.032, -0.032),
}


class FeaturePipeline:
    """Build full feature matrix for model training/prediction."""

    def __init__(
        self,
        use_news: bool = True,
        use_llm: bool = True,
        use_financials: bool = True,
    ):
        self.data_provider = get_provider(get_setting("data", "provider", default="yfinance"))
        self.use_news = use_news
        self.use_llm = use_llm
        self.use_financials = use_financials and bool(
            get_setting("features", "use_financials", default=True)
        )
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
        if self.use_financials:
            self.financial_generator = FinancialFeatureGenerator(
                announcement_lag_days=int(
                    get_setting("features", "financial_announcement_lag_days", default=45)
                )
            )

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
            logger.error(f"No price data returned by yfinance for {symbol} — check ticker format or network")
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

        # Step 5: Market context (NIFTY 50 relative strength)
        df = self._add_market_context(df, start_date, end_date)

        # Step 5b: Quarterly financial report features
        if self.use_financials:
            try:
                df = self.financial_generator.merge_financial_features(df, symbol)
            except Exception as e:
                logger.warning(f"Financial features failed for {symbol}: {e}")

        # Step 6: Create labels
        df = self._add_labels(df)

        # Replace inf values (e.g. from pct_change on zero volume) with NaN,
        # then drop all rows that still contain NaN or inf.
        df = df.replace([float("inf"), float("-inf")], float("nan"))
        df = df.dropna()

        logger.info(f"Built features for {symbol}: {df.shape}")
        return df

    def _add_market_context(
        self,
        df: pd.DataFrame,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        """Add NIFTY 50 market context features.

        Adds:
          NIFTY_Return_1d      — NIFTY's daily % return (market direction)
          Relative_Strength_1d — stock 1d return minus NIFTY 1d return
          Relative_Strength_5d — stock 5d return minus NIFTY 5d return
        """
        try:
            nifty_data = self.data_provider.fetch_historical("^NSEI", start_date, end_date)
            if nifty_data.is_empty:
                logger.warning("NIFTY data unavailable — skipping market context features")
                return df

            nifty_close = nifty_data.df[["Close"]].rename(columns={"Close": "_NIFTY"})
            df = df.join(nifty_close, how="left")

            nifty_1d = df["_NIFTY"].pct_change(1)
            nifty_5d = df["_NIFTY"].pct_change(5)

            df["NIFTY_Return_1d"]      = nifty_1d
            df["Relative_Strength_1d"] = df["Close"].pct_change(1) - nifty_1d
            df["Relative_Strength_5d"] = df["Close"].pct_change(5) - nifty_5d

            df = df.drop(columns=["_NIFTY"])
            logger.info("Added NIFTY market context features")
        except Exception as e:
            logger.warning(f"Market context features failed: {e} — continuing without them")

        return df

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return and signal labels.

        The signal is based on the configured prediction_horizon (days ahead).
        Thresholds are chosen from HORIZON_THRESHOLDS (horizon 1-10) so that
        the BUY/SELL band widens proportionally with expected volatility.
        return_1d and return_5d are always computed as reference columns but
        excluded from the feature matrix.
        """
        horizon = int(get_setting("features", "prediction_horizon", default=1))
        if horizon < 1 or horizon > 10:
            raise ValueError(
                f"prediction_horizon must be between 1 and 10, got {horizon}. "
                "Update config/settings.yaml."
            )

        buy_thresh, sell_thresh = HORIZON_THRESHOLDS[horizon]
        logger.info(
            f"Labelling with horizon={horizon}d, "
            f"buy>={buy_thresh:.1%}, sell<={sell_thresh:.1%}"
        )

        df = df.copy()
        df["return_1d"] = df["Close"].pct_change(1).shift(-1)
        df["return_5d"] = df["Close"].pct_change(5).shift(-5)

        # Compute the horizon return if it isn't one of the two above
        horizon_col = f"return_{horizon}d"
        if horizon_col not in df.columns:
            df[horizon_col] = df["Close"].pct_change(horizon).shift(-horizon)

        # Signal: 0=SELL, 1=HOLD, 2=BUY  — based on the configured horizon
        conditions = [
            df[horizon_col] <= sell_thresh,
            df[horizon_col] >= buy_thresh,
        ]
        choices = [0, 2]
        df["signal"] = np.select(conditions, choices, default=1)

        return df

    def prepare_regression_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Prepare regression data for encoder-decoder training.

        Targets are price ratios: close[t+k] / close[t] for k=1..horizon.
        The ratio is always positive and scale-invariant (1.0 = no change).

        Loop bound: ``range(seq_len, n_df - horizon)`` so that
        ``close_values[i + horizon]`` is always within the post-dropna array.

        Returns:
            (sequences, tabular, reg_targets, labels, feature_names)
            - sequences  : (N, seq_len, n_features)
            - tabular    : (N, n_features) — same features, row i
            - reg_targets: (N, horizon)   — future price ratios
            - labels     : (N,)           — classification signal (for reporting)
            - feature_names: list of feature column names
        """
        df = self.build_features(symbol, start_date, end_date)
        if df.empty:
            raise ValueError(
                "yfinance returned no price data — check ticker format or network"
            )

        horizon = int(get_setting("features", "prediction_horizon", default=1))
        min_rows = self.sequence_length + horizon + 10
        if len(df) < min_rows:
            raise ValueError(
                f"Only {len(df)} rows after feature build "
                f"(need {min_rows} for seq_len={self.sequence_length}, horizon={horizon}); "
                "use an earlier --start-date"
            )

        label_cols = {"return_1d", "return_5d", f"return_{horizon}d", "signal"}
        feature_cols = [c for c in df.columns if c not in label_cols]

        features = df[feature_cols].values          # (n_df, n_features)
        labels = df["signal"].values                # (n_df,)
        close_values = df["Close"].values           # (n_df,)
        n_df = len(features)

        sequences, tabular, reg_targets, seq_labels = [], [], [], []

        # i is the *current* row (end of seq window, before the horizon period)
        # We need close_values[i + horizon], so i < n_df - horizon
        for i in range(self.sequence_length, n_df - horizon):
            close_t = close_values[i]
            ratios = np.array(
                [close_values[i + k] / close_t for k in range(1, horizon + 1)],
                dtype=np.float32,
            )
            sequences.append(features[i - self.sequence_length : i])
            tabular.append(features[i])
            reg_targets.append(ratios)
            seq_labels.append(labels[i])

        sequences = np.array(sequences, dtype=np.float32)
        tabular = np.array(tabular, dtype=np.float32)
        reg_targets = np.array(reg_targets, dtype=np.float32)
        seq_labels = np.array(seq_labels, dtype=np.int64)

        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        tabular = np.nan_to_num(tabular, nan=0.0, posinf=0.0, neginf=0.0)
        reg_targets = np.nan_to_num(reg_targets, nan=1.0, posinf=1.0, neginf=1.0)

        logger.info(
            f"Prepared regression data for {symbol}: "
            f"sequences={sequences.shape}, reg_targets={reg_targets.shape}"
        )
        return sequences, tabular, reg_targets, seq_labels, feature_cols

    def prepare_prophet_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame | None, int]:
        """Extract date index, close prices, and features for Prophet.

        Two feature categories are included:
        - Lag-safe regressors (CANDIDATE_REGRESSORS): future values approximated
          by carry-forward.
        - Lagged regressors ({col}_lag{horizon}): col.shift(horizon).  At
          future step k, col_lag_h[t+k] = col[t+k-h] which is exactly known
          from historical data — no approximation needed.

        Returns:
            (dates, close_values, feature_df, n_df)
            - dates       : DatetimeIndex of all post-dropna rows
            - close_values: Close prices aligned with dates
            - feature_df  : DataFrame of lag-safe + lagged regressor columns,
                            or None if none are present
            - n_df        : total row count
        """
        from stock_prediction.models.prophet_model import (
            CANDIDATE_REGRESSORS,
            LAGGED_FEATURE_SOURCES,
        )

        horizon = int(get_setting("features", "prediction_horizon", default=1))

        df = self.build_features(symbol, start_date, end_date)
        if df.empty:
            raise ValueError(
                "yfinance returned no price data — check ticker format or network"
            )

        # Lag-safe columns (direct use, carry-forward for future)
        # Also include fin_* ratio columns (quarterly step functions — carry-
        # forward is correct between report dates) and report aging features
        # (carry-forward is a good approximation over a short 1-10d horizon).
        fin_cols = [
            c for c in df.columns
            if c.startswith("fin_")
            or c in {"report_age_days", "report_effect", "report_freshness",
                     "days_to_next_report"}
        ]
        lag_safe = list(dict.fromkeys(
            [c for c in CANDIDATE_REGRESSORS if c in df.columns] + fin_cols
        ))

        # Lagged columns: shift source column back by horizon days
        lag_cols: dict[str, pd.Series] = {}
        for src in LAGGED_FEATURE_SOURCES:
            if src in df.columns:
                lag_name = f"{src}_lag{horizon}"
                lag_cols[lag_name] = df[src].shift(horizon)

        if not lag_safe and not lag_cols:
            return df.index, df["Close"].values, None, len(df)

        feature_df = df[lag_safe].copy() if lag_safe else pd.DataFrame(index=df.index)
        for name, series in lag_cols.items():
            feature_df[name] = series

        # Drop rows with NaN introduced by lagging (first `horizon` rows)
        feature_df = feature_df.dropna()
        # Align df index to feature_df after dropna
        df_aligned_index = df.index.intersection(feature_df.index)
        feature_df = feature_df.loc[df_aligned_index]
        close_values = df.loc[df_aligned_index, "Close"].values
        dates = df_aligned_index

        logger.info(
            f"Prophet features for {symbol}: {list(feature_df.columns)} "
            f"({len(feature_df)} rows)"
        )
        return dates, close_values, feature_df, len(feature_df)

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
        if df.empty:
            raise ValueError(
                "yfinance returned no price data — check ticker format or network"
            )
        if len(df) < self.sequence_length + 10:
            raise ValueError(
                f"Only {len(df)} rows after feature build "
                f"(need {self.sequence_length + 10}); "
                f"use an earlier --start-date (default 2020-01-01 gives ~1200 rows)"
            )

        # Separate features and labels — exclude all future-looking return columns
        horizon = int(get_setting("features", "prediction_horizon", default=1))
        label_cols = {"return_1d", "return_5d", f"return_{horizon}d", "signal"}
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
