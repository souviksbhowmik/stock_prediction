"""Merge news features into price DataFrame."""

import pandas as pd

from stock_prediction.utils.logging import get_logger

logger = get_logger("features.news_based")


def merge_news_features(
    price_df: pd.DataFrame,
    news_features: dict[str, float],
) -> pd.DataFrame:
    """Add news feature columns to a price DataFrame.

    News features are constant for the latest day (applied to last row)
    and forward-filled for historical context.
    """
    df = price_df.copy()

    # Add news features as columns with the same value for all rows
    for key, value in news_features.items():
        if key.startswith("_"):
            continue  # Skip metadata keys like _summary
        df[f"news_{key}"] = float(value)

    return df


def merge_llm_features(
    price_df: pd.DataFrame,
    llm_scores: dict[str, float],
) -> pd.DataFrame:
    """Add LLM broker analysis features to a price DataFrame."""
    df = price_df.copy()

    for key, value in llm_scores.items():
        if key.startswith("_"):
            continue
        df[f"llm_{key}"] = float(value)

    return df
