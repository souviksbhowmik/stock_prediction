"""Tests for ML models."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from stock_prediction.models.lstm_model import StockLSTM, LSTMPredictor
from stock_prediction.models.xgboost_model import XGBoostPredictor
from stock_prediction.models.ensemble import EnsembleModel, EnsemblePrediction


class TestStockLSTM:
    def test_forward_pass(self):
        model = StockLSTM(input_size=10, hidden_size=32, num_layers=1)
        x = __import__("torch").randn(4, 60, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_logits(self):
        model = StockLSTM(input_size=5, hidden_size=16, num_layers=1)
        x = __import__("torch").randn(2, 30, 5)
        out = model(x)
        # Output should be raw logits (not probabilities)
        assert out.shape == (2, 3)


class TestLSTMPredictor:
    def test_train_and_predict(self, sample_features):
        sequences, tabular, labels = sample_features
        n_features = sequences.shape[2]

        predictor = LSTMPredictor(input_size=n_features)
        # Quick training with small epochs
        history = predictor.train(
            sequences[:80], labels[:80],
            sequences[80:], labels[80:],
        )
        assert "train_loss" in history

        probs = predictor.predict_proba(sequences[:5])
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_save_load(self, sample_features):
        sequences, _, labels = sample_features
        n_features = sequences.shape[2]

        predictor = LSTMPredictor(input_size=n_features)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            predictor.save(path)
            assert path.exists()

            new_predictor = LSTMPredictor(input_size=n_features)
            new_predictor.load(path)

            # Both should give same predictions
            p1 = predictor.predict_proba(sequences[:3])
            p2 = new_predictor.predict_proba(sequences[:3])
            np.testing.assert_array_almost_equal(p1, p2, decimal=4)


class TestXGBoostPredictor:
    def test_train_and_predict(self, sample_features):
        _, tabular, labels = sample_features

        predictor = XGBoostPredictor()
        predictor.train(tabular[:80], labels[:80], tabular[80:], labels[80:])

        probs = predictor.predict_proba(tabular[:5])
        assert probs.shape == (5, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance(self, sample_features):
        _, tabular, labels = sample_features
        names = [f"feature_{i}" for i in range(tabular.shape[1])]

        predictor = XGBoostPredictor()
        predictor.train(tabular[:80], labels[:80], feature_names=names)

        importance = predictor.get_feature_importance()
        assert len(importance) == tabular.shape[1]

    def test_save_load(self, sample_features):
        _, tabular, labels = sample_features

        predictor = XGBoostPredictor()
        predictor.train(tabular[:80], labels[:80])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            predictor.save(path)
            assert path.exists()

            new_predictor = XGBoostPredictor()
            new_predictor.load(path)

            p1 = predictor.predict_proba(tabular[:3])
            p2 = new_predictor.predict_proba(tabular[:3])
            np.testing.assert_array_almost_equal(p1, p2)


class TestEnsembleModel:
    def test_ensemble_predict(self, sample_features):
        sequences, tabular, labels = sample_features
        n_features = sequences.shape[2]

        lstm = LSTMPredictor(input_size=n_features)
        xgb = XGBoostPredictor()
        xgb.train(tabular[:80], labels[:80])

        ensemble = EnsembleModel(lstm, xgb)
        predictions = ensemble.predict(sequences[:5], tabular[:5])

        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, EnsemblePrediction)
            assert pred.signal in ("BUY", "HOLD", "SELL")
            assert 0 <= pred.confidence <= 1

    def test_predict_single(self, sample_features):
        sequences, tabular, labels = sample_features
        n_features = sequences.shape[2]

        lstm = LSTMPredictor(input_size=n_features)
        xgb = XGBoostPredictor()
        xgb.train(tabular[:80], labels[:80])

        ensemble = EnsembleModel(lstm, xgb)
        pred = ensemble.predict_single(sequences[0], tabular[0])

        assert isinstance(pred, EnsemblePrediction)
        assert "BUY" in pred.probabilities
        assert "SELL" in pred.probabilities
        assert "HOLD" in pred.probabilities
