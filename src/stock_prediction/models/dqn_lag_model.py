"""DQN agent trained on lag and trend features (fast alternative to GRU-DQN)."""
from stock_prediction.models.dqn_model import DQNPredictor


class DQNLagPredictor(DQNPredictor):
    """DQN agent whose state is the lag/trend feature vector.

    Uses the same 16-feature lag set as XGBoostLagPredictor:
    lagged close ratios, lagged indicator snapshots, ADX trend direction,
    EMA cross flag, normalised trend slopes, and price rank in recent range.

    Accepts 2-D tabular input (N, n_lag_features) — no sequence encoder
    needed because temporal context is encoded in the features themselves.
    Trains substantially faster than the sequence-encoder DQN.

    Saved as ``dqn_lag.pt``.
    """
    pass
