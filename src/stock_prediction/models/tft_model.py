"""Temporal Fusion Transformer for stock price ratio regression.

Architecture (simplified TFT following Lim et al. 2021):
  - Variable Selection Networks (VSN)   : soft per-timestep feature weighting
  - Gated Residual Networks  (GRN)      : gated skip-connection blocks
  - LSTM encoder                        : temporal state encoding
  - Multi-head self-attention           : interpretable temporal attention
  - Horizon output head                 : (B, horizon) price ratios

Same regression → Gaussian-CDF → probability approach as EncoderDecoderPredictor.
No new library dependencies — pure PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset

from stock_prediction.config import get_setting
from stock_prediction.features.pipeline import HORIZON_THRESHOLDS
from stock_prediction.utils.logging import get_logger

logger = get_logger("models.tft")


# ---------------------------------------------------------------------------
# TFT building blocks
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network — core TFT building block.

    Output = LayerNorm(skip(x) + GLU(ELU(Linear(x))))

    Args:
        input_size  : dimension of input
        hidden_size : internal projection size
        output_size : dimension of output
        dropout     : dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size * 2)   # for GLU gating
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        # Skip connection projects only when dimensions differ
        self.skip = (
            nn.Linear(input_size, output_size, bias=False)
            if input_size != output_size
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)                          # (..., output_size*2)
        h_gate, h_val = h.chunk(2, dim=-1)       # each (..., output_size)
        h = h_val * torch.sigmoid(h_gate)        # Gated Linear Unit
        h = self.dropout(h)
        return self.norm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — learns per-timestep feature importance.

    Applies one GRN per feature, then computes a softmax weighting
    over all features and returns the weighted sum.

    Args:
        n_features  : number of input features
        hidden_size : GRN hidden dimension
        dropout     : dropout probability
    """

    def __init__(self, n_features: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        # One GRN per feature: scalar → hidden_size
        self.feature_grns = nn.ModuleList(
            [GatedResidualNetwork(1, hidden_size, hidden_size, dropout)
             for _ in range(n_features)]
        )
        # Selector GRN: n_features → n_features (for softmax weights)
        self.selector = GatedResidualNetwork(n_features, hidden_size, n_features, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., n_features)

        Returns:
            selected : (..., hidden_size) — weighted combination
            weights  : (..., n_features)  — softmax importance weights
        """
        # Process each feature independently
        processed = torch.stack(
            [self.feature_grns[i](x[..., i : i + 1]) for i in range(self.n_features)],
            dim=-1,
        )  # (..., hidden_size, n_features)

        # Softmax weights over features
        weights = F.softmax(self.selector(x), dim=-1)  # (..., n_features)

        # Weighted sum across features
        selected = (processed * weights.unsqueeze(-2)).sum(dim=-1)  # (..., hidden_size)
        return selected, weights


class TemporalFusionTransformer(nn.Module):
    """Simplified TFT for multi-step regression.

    Data flow:
      1. Variable Selection Network per timestep → (B, T, hidden)
      2. LSTM encoder                            → (B, T, hidden)
      3. Multi-head self-attention               → (B, T, hidden)
      4. Post-attention GRN                      → (B, T, hidden)
      5. Output projection on last timestep      → (B, horizon)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size

        # Ensure hidden_size is divisible by num_heads
        if hidden_size % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads

        # 1. Variable Selection Network (applied per-timestep)
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)

        # 2. LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # 3. Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # 4. Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # 5. Output projection
        self.output_proj = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)

        Returns:
            (B, horizon) predicted price ratios
        """
        B, T, F = x.shape

        # ── 1. Variable Selection (flatten to (B*T, F), then restore) ──────
        x_flat = x.reshape(B * T, F)
        selected, _ = self.vsn(x_flat)              # (B*T, hidden)
        selected = selected.reshape(B, T, -1)        # (B, T, hidden)

        # ── 2. LSTM encoding ───────────────────────────────────────────────
        lstm_out, _ = self.lstm(selected)            # (B, T, hidden)

        # ── 3. Multi-head self-attention ───────────────────────────────────
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(lstm_out + attn_out)  # residual + norm

        # ── 4. Post-attention GRN ──────────────────────────────────────────
        refined = self.post_attn_grn(attn_out)       # (B, T, hidden)

        # ── 5. Output on last timestep ─────────────────────────────────────
        last = refined[:, -1, :]                     # (B, hidden)
        return self.output_proj(last)                # (B, horizon)


# ---------------------------------------------------------------------------
# Predictor wrapper (mirrors EncoderDecoderPredictor API exactly)
# ---------------------------------------------------------------------------

class TFTPredictor:
    """Wrapper around TemporalFusionTransformer for training and inference.

    Targets are *price ratios*: close[t+k] / close[t] for k=1..horizon.
    Ratio=1.0 → no change; ratio−1 is the return.

    Tuning metric  : MAPE on the final-step ratio (most actionable).
    Reporting metric: balanced accuracy after binning predictions → SELL/HOLD/BUY.
    Probabilities  : Gaussian CDF model using residual std estimated from val.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        num_heads: int | None = None,
        num_lstm_layers: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        patience: int | None = None,
        horizon: int | None = None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else int(
            get_setting("models", "tft", "hidden_size", default=128)
        )
        self.num_heads = num_heads if num_heads is not None else int(
            get_setting("models", "tft", "num_heads", default=4)
        )
        self.num_lstm_layers = num_lstm_layers if num_lstm_layers is not None else int(
            get_setting("models", "tft", "num_lstm_layers", default=2)
        )
        self.dropout = dropout if dropout is not None else float(
            get_setting("models", "tft", "dropout", default=0.1)
        )
        self.lr = learning_rate if learning_rate is not None else float(
            get_setting("models", "tft", "learning_rate", default=0.001)
        )
        self.epochs = epochs if epochs is not None else int(
            get_setting("models", "tft", "epochs", default=50)
        )
        self.batch_size = batch_size if batch_size is not None else int(
            get_setting("models", "tft", "batch_size", default=32)
        )
        self.patience = patience if patience is not None else int(
            get_setting("models", "tft", "patience", default=10)
        )
        self.horizon = horizon if horizon is not None else int(
            get_setting("features", "prediction_horizon", default=1)
        )

        self.device = self._get_device()
        self.model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_lstm_layers=self.num_lstm_layers,
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
    ) -> dict:
        """Train with MSE loss on ratio targets; early stop on val MSE.

        After training, estimates residual_std from val predictions for
        use in the Gaussian CDF probability model.
        """
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        history: dict = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None
        best_epoch = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.model(X_batch)              # (B, horizon)
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
                    logger.info(f"TFT early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"TFT epoch {epoch+1}/{self.epochs} — "
                        f"train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"TFT epoch {epoch+1}/{self.epochs} — loss={avg_loss:.6f}"
                    )

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
        logger.info(f"TFT residual_std={self._residual_std:.6f}")

    # ── Evaluation ────────────────────────────────────────────────────────

    def compute_mape(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """MAPE on the final-step prediction (the actionable signal step)."""
        ratios = self.predict_ratios(X_val)
        pred_last = ratios[:, -1]
        true_last = y_val[:, -1]
        return float(
            np.mean(np.abs((true_last - pred_last) / np.maximum(np.abs(true_last), 1e-8)))
        )

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_ratios(self, X: np.ndarray) -> np.ndarray:
        """Return predicted price ratios, shape (N, horizon)."""
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.FloatTensor(X[start : start + self.batch_size]).to(self.device)
                out = self.model(batch)
                outputs.append(out.cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert TFT regression output to class probabilities (N, 3).

        Uses a Gaussian CDF model centred on the predicted final-step return
        with std=residual_std estimated from validation.

        Class order: 0=SELL, 1=HOLD, 2=BUY
        """
        ratios = self.predict_ratios(X)            # (N, horizon)
        pred_return = ratios[:, -1] - 1.0          # final-step return

        buy_thresh, sell_thresh = self._thresholds
        sigma = self._residual_std

        p_sell = norm.cdf(sell_thresh, loc=pred_return, scale=sigma)
        p_buy  = 1.0 - norm.cdf(buy_thresh, loc=pred_return, scale=sigma)
        p_hold = np.clip(1.0 - p_sell - p_buy, 0.0, 1.0)

        probs = np.stack([p_sell, p_hold, p_buy], axis=1)   # (N, 3)
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
                "model_state":     self.model.state_dict(),
                "input_size":      self.input_size,
                "hidden_size":     self.hidden_size,
                "num_heads":       self.num_heads,
                "num_lstm_layers": self.num_lstm_layers,
                "dropout":         self.dropout,
                "horizon":         self.horizon,
                "residual_std":    self._residual_std,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.hidden_size     = checkpoint.get("hidden_size", self.hidden_size)
        self.num_heads       = checkpoint.get("num_heads", self.num_heads)
        self.num_lstm_layers = checkpoint.get("num_lstm_layers", self.num_lstm_layers)
        self.dropout         = checkpoint.get("dropout", self.dropout)
        self.horizon         = checkpoint.get("horizon", self.horizon)
        self._residual_std   = checkpoint.get("residual_std", self._residual_std)
        self.model = TemporalFusionTransformer(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
