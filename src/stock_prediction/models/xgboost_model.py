"""XGBoost model for stock prediction."""

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.xgboost")


class XGBoostPredictor:
    """XGBoost classifier for stock signal prediction."""

    def __init__(self):
        self.n_estimators = get_setting("models", "xgboost", "n_estimators", default=500)
        self.max_depth = get_setting("models", "xgboost", "max_depth", default=6)
        self.learning_rate = get_setting("models", "xgboost", "learning_rate", default=0.05)
        self.early_stopping_rounds = get_setting(
            "models", "xgboost", "early_stopping_rounds", default=20
        )
        self.subsample = get_setting("models", "xgboost", "subsample", default=0.8)
        self.colsample_bytree = get_setting("models", "xgboost", "colsample_bytree", default=0.8)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
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
    ) -> dict:
        """Train the XGBoost model."""
        if feature_names:
            self.feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Get best iteration info
        results = self.model.evals_result()
        train_loss = results["validation_0"]["mlogloss"]
        val_loss = results.get("validation_1", {}).get("mlogloss", [])

        logger.info(
            f"XGBoost trained: {self.model.best_iteration} iterations, "
            f"best score: {self.model.best_score:.4f}"
        )
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
