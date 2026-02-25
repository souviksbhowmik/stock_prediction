"""Encoder-Decoder LSTM for stock price ratio regression."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import HORIZON_THRESHOLDS
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.encoder_decoder")


class EncoderDecoderLSTM(nn.Module):
    """Seq2seq LSTM that predicts a sequence of future price ratios.

    Encoder  — processes the 60-timestep feature sequence → hidden state.
    Decoder  — auto-regressively generates *horizon* future price ratios
               (close[t+k] / close[t]) starting from ratio=1.0 as the
               initial token, with optional teacher forcing during training.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Decoder input: single scalar ratio per step
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            x       : (B, seq_len, input_size) — input feature sequences
            targets : (B, horizon) — ground-truth ratios for teacher forcing
            teacher_forcing_ratio: prob of using ground-truth as decoder input

        Returns:
            (B, horizon) — predicted price ratios
        """
        B = x.size(0)
        _, (h_n, c_n) = self.encoder(x)            # encode full sequence

        # Initial decoder token: ratio=1.0 (no-change baseline)
        dec_input = torch.ones(B, 1, 1, device=x.device)  # (B, 1, 1)

        outputs: list[torch.Tensor] = []
        for k in range(self.horizon):
            dec_out, (h_n, c_n) = self.decoder(dec_input, (h_n, c_n))  # (B,1,H)
            ratio_k = self.output_proj(self.dropout(dec_out))           # (B,1,1)
            outputs.append(ratio_k.squeeze(-1))                         # (B,1)

            # Teacher forcing: use ground truth or own prediction as next input
            if targets is not None and random.random() < teacher_forcing_ratio:
                dec_input = targets[:, k : k + 1].unsqueeze(-1)        # (B,1,1) GT
            else:
                dec_input = ratio_k.detach()                            # (B,1,1)

        return torch.cat(outputs, dim=1)  # (B, horizon)


class EncoderDecoderPredictor:
    """Wrapper around EncoderDecoderLSTM for training and inference.

    Targets are *price ratios* — close[t+k] / close[t] for k=1..horizon.
    Ratio=1.0 means no change; ratio-1 is the return.

    Tuning metric  : MAPE on the final-step ratio (most predictive for signal).
    Reporting metric: balanced accuracy after binning predictions → SELL/HOLD/BUY.
    Probabilities  : Gaussian CDF model using residual std estimated from val.
    """

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
        horizon: int | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else int(
            get_setting("models", "encoder_decoder", "hidden_size", default=128)
        )
        self.num_layers = num_layers if num_layers is not None else int(
            get_setting("models", "encoder_decoder", "num_layers", default=2)
        )
        self.dropout = dropout if dropout is not None else float(
            get_setting("models", "encoder_decoder", "dropout", default=0.3)
        )
        self.lr = learning_rate if learning_rate is not None else float(
            get_setting("models", "encoder_decoder", "learning_rate", default=0.001)
        )
        self.epochs = epochs if epochs is not None else int(
            get_setting("models", "encoder_decoder", "epochs", default=50)
        )
        self.batch_size = batch_size if batch_size is not None else int(
            get_setting("models", "encoder_decoder", "batch_size", default=32)
        )
        self.patience = patience if patience is not None else int(
            get_setting("models", "encoder_decoder", "patience", default=10)
        )
        self.horizon = horizon if horizon is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )

        self.device = self._get_device()
        self.model = EncoderDecoderLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)

        # Estimated from validation residuals after training
        self._residual_std: float = 0.02
        # Horizon-specific thresholds for buy/sell classification
        self._thresholds: tuple[float, float] = HORIZON_THRESHOLDS.get(
            self.horizon, (0.022, -0.022)
        )

    # ── Device ────────────────────────────────────────────────────────────

    def _get_device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,          # (N_train, seq_len, n_features)
        y_train: np.ndarray,          # (N_train, horizon) — ratio targets
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> dict:
        """Train with MSE loss on ratio targets; early stop on val MSE.

        After training, estimates residual_std from val predictions for
        use in the Gaussian CDF probability model.
        """
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        history: dict = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch, targets=y_batch,
                                   teacher_forcing_ratio=teacher_forcing_ratio)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            if X_val is not None and y_val is not None:
                val_loss = self._mse_loss(X_val, y_val)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"ED early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"ED epoch {epoch+1}/{self.epochs} — "
                        f"train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"ED epoch {epoch+1}/{self.epochs} — loss={avg_loss:.6f}")

        history["best_epoch"] = best_epoch if best_epoch > 0 else (epoch + 1)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        if X_val is not None and y_val is not None:
            self._estimate_residual_std(X_val, y_val)

        return history

    def _mse_loss(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_val).to(self.device)
            y_t = torch.FloatTensor(y_val).to(self.device)
            loss = nn.MSELoss()(self.model(X_t), y_t)
        return float(loss.item())

    def _estimate_residual_std(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Estimate residual std from final-step val predictions."""
        ratios = self.predict_ratios(X_val)
        residuals = y_val[:, -1] - ratios[:, -1]
        self._residual_std = max(float(np.std(residuals)), 1e-4)
        logger.info(f"ED residual_std={self._residual_std:.6f}")

    # ── Evaluation ────────────────────────────────────────────────────────

    def compute_mape(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """MAPE on the final-step prediction (the actionable signal step)."""
        ratios = self.predict_ratios(X_val)
        pred_last = ratios[:, -1]
        true_last = y_val[:, -1]
        return float(np.mean(np.abs((true_last - pred_last) /
                                    np.maximum(np.abs(true_last), 1e-8))))

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_ratios(self, X: np.ndarray) -> np.ndarray:
        """Return predicted price ratios, shape (N, horizon)."""
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.FloatTensor(X[start : start + self.batch_size]).to(self.device)
                out = self.model(batch)   # no teacher forcing at inference
                outputs.append(out.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert ED regression output to class probabilities (N, 3).

        Uses a Gaussian CDF model centred on the predicted final-step return
        with std=residual_std estimated from validation.

        Class order: 0=SELL, 1=HOLD, 2=BUY
        """
        ratios = self.predict_ratios(X)            # (N, horizon)
        pred_return = ratios[:, -1] - 1.0          # final-step return

        buy_thresh, sell_thresh = self._thresholds
        sigma = self._residual_std

        p_sell = norm.cdf(sell_thresh, loc=pred_return, scale=sigma)
        p_buy = 1.0 - norm.cdf(buy_thresh, loc=pred_return, scale=sigma)
        p_hold = np.clip(1.0 - p_sell - p_buy, 0.0, 1.0)

        probs = np.stack([p_sell, p_hold, p_buy], axis=1)          # (N, 3)
        row_sums = probs.sum(axis=1, keepdims=True)
        return (probs / np.maximum(row_sums, 1e-8)).astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels (0=SELL, 1=HOLD, 2=BUY)."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "horizon": self.horizon,
                "residual_std": self._residual_std,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.hidden_size = checkpoint.get("hidden_size", self.hidden_size)
        self.num_layers = checkpoint.get("num_layers", self.num_layers)
        self.dropout = checkpoint.get("dropout", self.dropout)
        self.horizon = checkpoint.get("horizon", self.horizon)
        self._residual_std = checkpoint.get("residual_std", self._residual_std)
        self.model = EncoderDecoderLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
