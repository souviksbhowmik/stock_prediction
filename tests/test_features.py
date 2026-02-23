"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np

from stock_prediction.features.news_based import merge_news_features, merge_llm_features


def test_merge_news_features(sample_ohlcv_df):
    news_features = {
        "sentiment_1d_mean": 0.5,
        "sentiment_7d_mean": 0.3,
        "news_volume_1d": 5.0,
        "_summary": "should be skipped",
    }
    df = merge_news_features(sample_ohlcv_df, news_features)

    assert "news_sentiment_1d_mean" in df.columns
    assert "news_sentiment_7d_mean" in df.columns
    assert "news_news_volume_1d" in df.columns
    assert "news__summary" not in df.columns  # underscore prefix skipped


def test_merge_llm_features(sample_ohlcv_df):
    llm_scores = {
        "earnings_outlook": 7.0,
        "overall_broker_score": 6.5,
        "_summary": "skip this",
    }
    df = merge_llm_features(sample_ohlcv_df, llm_scores)

    assert "llm_earnings_outlook" in df.columns
    assert "llm_overall_broker_score" in df.columns
    assert "llm__summary" not in df.columns


def test_merge_preserves_index(sample_ohlcv_df):
    df = merge_news_features(sample_ohlcv_df, {"sentiment_1d_mean": 0.5})
    assert len(df) == len(sample_ohlcv_df)
    assert df.index.equals(sample_ohlcv_df.index)


def test_merge_does_not_modify_original(sample_ohlcv_df):
    original_cols = set(sample_ohlcv_df.columns)
    merge_news_features(sample_ohlcv_df, {"test": 1.0})
    assert set(sample_ohlcv_df.columns) == original_cols
