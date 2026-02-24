"""LSTM model for stock prediction."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.lstm")


class StockLSTM(nn.Module):
    """2-layer LSTM for stock signal classification."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class LSTMPredictor:
    """Wrapper for training and prediction with StockLSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else get_setting("models", "lstm", "hidden_size", default=128)
        self.num_layers = num_layers if num_layers is not None else get_setting("models", "lstm", "num_layers", default=2)
        self.dropout = dropout if dropout is not None else get_setting("models", "lstm", "dropout", default=0.3)
        self.lr = learning_rate if learning_rate is not None else get_setting("models", "lstm", "learning_rate", default=0.001)
        self.epochs = epochs if epochs is not None else get_setting("models", "lstm", "epochs", default=50)
        self.batch_size = batch_size if batch_size is not None else get_setting("models", "lstm", "batch_size", default=32)
        self.patience = patience if patience is not None else get_setting("models", "lstm", "patience", default=10)

        self.device = self._get_device()
        self.model = StockLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        class_weights: np.ndarray | None = None,
    ) -> dict[str, list[float] | int]:
        """Train the LSTM model.

        Args:
            class_weights: Per-class weights shape (num_classes,) to counter
                           class imbalance. Pass None to use uniform weights.
        """
        self.model.train()
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if class_weights is not None:
            weight_tensor = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        history: dict[str, list[float] | int] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_epoch = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)  # type: ignore[union-attr]

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, criterion)
                history["val_loss"].append(val_loss)  # type: ignore[union-attr]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"train_loss: {avg_loss:.4f}, val_loss: {val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - loss: {avg_loss:.4f}")

        history["best_epoch"] = best_epoch if best_epoch > 0 else (epoch + 1)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return history

    def _evaluate(self, X: np.ndarray, y: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.LongTensor(y).to(self.device)
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
        self.model.train()
        return loss.item()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns shape (N, 3)."""
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }, path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # Rebuild model with the saved architecture before loading weights.
        # This handles cases where tuned hyperparams differ from config defaults.
        self.hidden_size = checkpoint.get("hidden_size", self.hidden_size)
        self.num_layers = checkpoint.get("num_layers", self.num_layers)
        self.dropout = checkpoint.get("dropout", self.dropout)
        self.model = StockLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
