"""XGBoost model for stock prediction."""

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.xgboost")

_UNSET = object()  # sentinel to distinguish "not provided" from explicit None


class XGBoostPredictor:
    """XGBoost classifier for stock signal prediction."""

    def __init__(
        self,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        learning_rate: float | None = None,
        early_stopping_rounds: int | None = _UNSET,  # type: ignore[assignment]
        subsample: float | None = None,
        colsample_bytree: float | None = None,
    ):
        self.n_estimators = n_estimators if n_estimators is not None else get_setting("models", "xgboost", "n_estimators", default=500)
        self.max_depth = max_depth if max_depth is not None else get_setting("models", "xgboost", "max_depth", default=6)
        self.learning_rate = learning_rate if learning_rate is not None else get_setting("models", "xgboost", "learning_rate", default=0.05)
        self.early_stopping_rounds = (
            get_setting("models", "xgboost", "early_stopping_rounds", default=20)
            if early_stopping_rounds is _UNSET
            else early_stopping_rounds
        )
        self.subsample = subsample if subsample is not None else get_setting("models", "xgboost", "subsample", default=0.8)
        self.colsample_bytree = colsample_bytree if colsample_bytree is not None else get_setting("models", "xgboost", "colsample_bytree", default=0.8)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=self.early_stopping_rounds,
            verbosity=0,
        )
        self.feature_names: list[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict:
        """Train the XGBoost model.

        Args:
            sample_weight: Per-sample weights to counter class imbalance.
                           Pass None to use uniform weights.
        """
        if feature_names:
            self.feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=False,
        )

        # Get best iteration info
        results = self.model.evals_result()
        train_loss = results["validation_0"]["mlogloss"]
        val_loss = results.get("validation_1", {}).get("mlogloss", [])

        best_iter = getattr(self.model, "best_iteration", self.n_estimators)
        best_score = getattr(self.model, "best_score", None)
        score_str = f", best score: {best_score:.4f}" if best_score is not None else ""
        logger.info(f"XGBoost trained: {best_iter} iterations{score_str}")
        return {"train_loss": train_loss, "val_loss": val_loss}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns shape (N, 3)."""
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return {f"f{i}": v for i, v in enumerate(importance)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, path)

    def load(self, path: str | Path) -> None:
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names", [])
