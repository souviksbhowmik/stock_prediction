"""Tests for model staleness warning feature."""

from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import joblib
import pytest
from sklearn.preprocessing import StandardScaler

from stock_prediction.models.trainer import ModelTrainer


@pytest.fixture
def model_dir(tmp_path):
    """Create a minimal model directory with required files."""
    symbol_dir = tmp_path / "TEST_NS"
    symbol_dir.mkdir()

    # Create dummy lstm and xgboost files
    (symbol_dir / "lstm.pt").write_bytes(b"dummy")
    (symbol_dir / "xgboost.joblib").write_bytes(b"dummy")

    return symbol_dir


def _save_meta(symbol_dir, trained_at=None):
    """Save a meta.joblib with optional trained_at timestamp."""
    meta = {
        "scaler": StandardScaler(),
        "seq_scaler": StandardScaler(),
        "feature_names": ["f1", "f2"],
        "input_size": 2,
    }
    if trained_at is not None:
        meta["trained_at"] = trained_at
    joblib.dump(meta, symbol_dir / "meta.joblib")


@patch.object(ModelTrainer, "__init__", lambda self, **kw: None)
class TestStaleness:
    """Tests for trained_at timestamp and model age detection."""

    def _make_trainer(self, save_dir):
        trainer = ModelTrainer()
        trainer.save_dir = save_dir
        return trainer

    @patch("stock_prediction.models.trainer.LSTMPredictor")
    @patch("stock_prediction.models.trainer.XGBoostPredictor")
    def test_load_returns_age_days(self, mock_xgb_cls, mock_lstm_cls, model_dir):
        """load_models returns correct age in days."""
        trained_at = (datetime.now() - timedelta(days=10)).isoformat()
        _save_meta(model_dir, trained_at=trained_at)

        mock_lstm_cls.return_value = MagicMock()
        mock_xgb_cls.return_value = MagicMock()

        trainer = self._make_trainer(model_dir.parent)
        _, _, _, model_age_days = trainer.load_models("TEST.NS")

        assert model_age_days is not None
        assert 9 <= model_age_days <= 11  # allow for clock drift

    @patch("stock_prediction.models.trainer.LSTMPredictor")
    @patch("stock_prediction.models.trainer.XGBoostPredictor")
    def test_load_old_model_no_trained_at(self, mock_xgb_cls, mock_lstm_cls, model_dir):
        """Old models without trained_at return None age (no crash)."""
        _save_meta(model_dir, trained_at=None)

        mock_lstm_cls.return_value = MagicMock()
        mock_xgb_cls.return_value = MagicMock()

        trainer = self._make_trainer(model_dir.parent)
        _, _, _, model_age_days = trainer.load_models("TEST.NS")

        assert model_age_days is None

    @patch("stock_prediction.models.trainer.LSTMPredictor")
    @patch("stock_prediction.models.trainer.XGBoostPredictor")
    def test_load_fresh_model_zero_days(self, mock_xgb_cls, mock_lstm_cls, model_dir):
        """A model trained just now should have age 0."""
        _save_meta(model_dir, trained_at=datetime.now().isoformat())

        mock_lstm_cls.return_value = MagicMock()
        mock_xgb_cls.return_value = MagicMock()

        trainer = self._make_trainer(model_dir.parent)
        _, _, _, model_age_days = trainer.load_models("TEST.NS")

        assert model_age_days == 0

    def test_save_includes_trained_at(self, tmp_path):
        """_save_models writes trained_at to meta.joblib."""
        trainer = self._make_trainer(tmp_path)

        lstm = MagicMock()
        lstm.input_size = 5
        xgb = MagicMock()
        scaler = StandardScaler()
        seq_scaler = StandardScaler()

        trainer._save_models("TEST.NS", lstm, xgb, scaler, seq_scaler, ["f1"])

        meta_path = tmp_path / "TEST_NS" / "meta.joblib"
        assert meta_path.exists()

        meta = joblib.load(meta_path)
        assert "trained_at" in meta
        # Should be a valid ISO timestamp
        dt = datetime.fromisoformat(meta["trained_at"])
        assert (datetime.now() - dt).total_seconds() < 60
