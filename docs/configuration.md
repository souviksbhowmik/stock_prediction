# Configuration Reference

All configuration lives in **`config/settings.yaml`**. Changes take effect immediately on the next run — no code changes required.

---

## Table of Contents

1. [data](#1-data)
2. [news](#2-news)
3. [sentiment](#3-sentiment)
4. [ner](#4-ner)
5. [llm](#5-llm)
6. [features](#6-features)
7. [models](#7-models)
8. [signals](#8-signals)
9. [screener](#9-screener)
10. [report](#10-report)
11. [paper_trading](#11-paper_trading)

---

## 1. `data`

Controls how historical price data is fetched.

```yaml
data:
  provider: yfinance
  default_start_date: "2020-01-01"
  cache_dir: data/raw
```

| Key | Default | Description |
|---|---|---|
| `provider` | `yfinance` | Data provider. Currently only `yfinance` is supported. |
| `default_start_date` | `2020-01-01` | Start date used when `--start-date` is not passed on the CLI. Earlier dates give more training data (~1200 rows from 2020). |
| `cache_dir` | `data/raw` | Directory where downloaded OHLCV CSVs are cached. |

---

## 2. `news`

Controls Google News RSS fetching used for sentiment and trending analysis.

```yaml
news:
  rss_base_url: "https://news.google.com/rss/search"
  queries:
    - "Indian stock market"
    - "NSE BSE stocks"
    - "NIFTY 50"
    - "Indian economy"
  max_articles_per_query: 50
  cache_dir: data/news_cache
  cache_expiry_hours: 6
```

| Key | Default | Description |
|---|---|---|
| `rss_base_url` | Google News RSS | Base URL for RSS queries. Do not change unless proxying. |
| `queries` | 4 market-level queries | Search terms fetched on every news run. Add stock-specific terms here for broader coverage. |
| `max_articles_per_query` | `50` | Maximum articles fetched per query. Higher values increase coverage but slow down runs. |
| `cache_dir` | `data/news_cache` | Directory for cached RSS responses. |
| `cache_expiry_hours` | `6` | How long cached news is considered fresh. Set to `0` to always fetch live. |

---

## 3. `sentiment`

Controls the FinBERT sentiment model used to score news headlines.

```yaml
sentiment:
  model_name: "ProsusAI/finbert"
  batch_size: 32
```

| Key | Default | Description |
|---|---|---|
| `model_name` | `ProsusAI/finbert` | HuggingFace model ID. FinBERT is fine-tuned on financial text and produces positive/negative/neutral scores. |
| `batch_size` | `32` | Number of headlines processed per inference batch. Reduce if running out of RAM/VRAM. |

---

## 4. `ner`

Controls the spaCy Named Entity Recognition model used to extract company names from news text.

```yaml
ner:
  spacy_model: "en_core_web_sm"
```

| Key | Default | Description |
|---|---|---|
| `spacy_model` | `en_core_web_sm` | spaCy model for NER. `en_core_web_sm` is fast and lightweight. Use `en_core_web_lg` for higher accuracy at the cost of ~750 MB extra disk space. |

---

## 5. `llm`

Controls the local LLM used for broker-style news analysis (overall scores, earnings outlook, sector momentum, etc.).

```yaml
llm:
  provider: ollama
  ollama:
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"
    timeout: 120
  cache_dir: data/news_cache
  cache_expiry_hours: 24
```

| Key | Default | Description |
|---|---|---|
| `provider` | `ollama` | LLM backend. Currently only `ollama` is supported. |
| `ollama.model` | `qwen2.5:7b` | Ollama model tag. Must be pulled first (`ollama pull <model>`). See model options below. |
| `ollama.base_url` | `http://localhost:11434` | Ollama server address. Change if running Ollama on a remote machine. |
| `ollama.timeout` | `120` | Seconds to wait for a single LLM response before timing out. Increase for slower hardware or larger models. |
| `cache_dir` | `data/news_cache` | Directory for cached LLM responses (shared with news cache). |
| `cache_expiry_hours` | `24` | How long cached LLM analysis is reused. LLM calls are slow, so 24 hours is a sensible default. |

**Model options for `ollama.model`:**

| Model | RAM needed | JSON reliability | Recommended for |
|---|---|---|---|
| `qwen2.5:7b` ← default | ~8 GB | Excellent | Best structured JSON output for broker scores |
| `llama3.1:8b` | ~8 GB | Good | Strong general model, slightly more natural prose |
| `llama3.2:3b` | ~4 GB | Fair | Low-RAM machines; lower quality broker scores |
| `llama3.1:70b` | ~48 GB | Best | High-end GPU setups only |

Pull a model before use: `ollama pull qwen2.5:7b`

**Note:** LLM features are optional. Pass `--no-llm` on the CLI to skip all LLM calls.

---

## 6. `features`

Controls feature engineering for model training and prediction.

```yaml
features:
  technical:
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    bb_period: 20
    bb_std: 2
    sma_periods: [20, 50]
    ema_periods: [12, 26]
    atr_period: 14
    stoch_period: 14
  sequence_length: 60
  prediction_horizon: 1
```

### `features.technical`

Standard technical indicator parameters. These follow widely-used defaults; change them only if you have a specific reason.

| Key | Default | Description |
|---|---|---|
| `rsi_period` | `14` | Lookback window for Relative Strength Index. |
| `macd_fast` | `12` | Fast EMA period for MACD. |
| `macd_slow` | `26` | Slow EMA period for MACD. |
| `macd_signal` | `9` | Signal line EMA period for MACD. |
| `bb_period` | `20` | Bollinger Bands moving average window. |
| `bb_std` | `2` | Number of standard deviations for Bollinger Band width. |
| `sma_periods` | `[20, 50]` | List of Simple Moving Average windows computed as features. |
| `ema_periods` | `[12, 26]` | List of Exponential Moving Average windows computed as features. |
| `atr_period` | `14` | Average True Range lookback (volatility measure). |
| `stoch_period` | `14` | Stochastic Oscillator lookback window. |

### `features.sequence_length`

| Key | Default | Description |
|---|---|---|
| `sequence_length` | `60` | Number of past trading days fed into the LSTM as one input sequence (~3 months). XGBoost uses only the current day's snapshot. Increasing this requires proportionally more historical data. |

### `features.prediction_horizon`

| Key | Default | Description |
|---|---|---|
| `prediction_horizon` | `5` | How many trading days ahead the signal label is based on. The label is a **point-to-point return** (day `t` close vs day `t+horizon` close), not an average over the period. Changing this requires retraining all models. |

**Horizon options:**

| Value | Meaning | Suitable for |
|---|---|---|
| `1` | Next trading day | Short-term / intraday-style signals |
| `5` | Same day next week ← default | Weekly swing trading (lands on same weekday) |
| `7` | ~2 calendar weeks | Medium-term swing trading |
| `10` | ~2 calendar weeks | Positional trades |
| `21` | ~1 calendar month | Long-term position building |

---

## 7. `models`

Controls model architecture and training behaviour.

```yaml
models:
  lstm:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    learning_rate: 0.001
    epochs: 50
    batch_size: 32
    patience: 10
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    early_stopping_rounds: 20
    subsample: 0.8
    colsample_bytree: 0.8
  ensemble:
    lstm_weight: 0.4
    xgboost_weight: 0.6
  save_dir: data/models
  train_split: 0.8
  staleness_warning_days: 30
```

### `models.lstm`

These are the **search space defaults** used by the hyperparameter tuner. The tuner will try combinations around these values and pick the best; the winner is then used for the full-data retrain.

| Key | Default | Description |
|---|---|---|
| `hidden_size` | `128` | Number of LSTM hidden units per layer. Tuner searches `[64, 128]`. |
| `num_layers` | `2` | Number of stacked LSTM layers. Not tuned; change here to fix for all runs. |
| `dropout` | `0.3` | Dropout rate applied between LSTM layers and before the output head. Tuner searches `[0.2, 0.3]`. |
| `learning_rate` | `0.001` | Adam optimiser learning rate. Tuner uses `0.001` as its fixed value. |
| `epochs` | `50` | Maximum training epochs. Early stopping (via `patience`) usually stops before this. |
| `batch_size` | `32` | Mini-batch size for gradient updates. |
| `patience` | `10` | Early stopping patience — stops training if val loss does not improve for this many consecutive epochs. |

### `models.xgboost`

These are the **search space defaults** for the XGBoost tuner.

| Key | Default | Description |
|---|---|---|
| `n_estimators` | `500` | Maximum number of boosting trees. Actual count is determined by early stopping during tuning; the final full-data model scales this up by `1 / train_split`. |
| `max_depth` | `6` | Maximum tree depth. Tuner searches `[4, 6]`. Deeper trees capture more complex patterns but overfit more easily. |
| `learning_rate` | `0.05` | Shrinkage applied to each tree's contribution. Tuner searches `[0.05, 0.10]`. Lower values need more trees. |
| `early_stopping_rounds` | `20` | Stop boosting if val loss does not improve for this many rounds during tuning. Disabled (`None`) for the final full-data retrain. |
| `subsample` | `0.8` | Fraction of training rows sampled per tree. Reduces overfitting. |
| `colsample_bytree` | `0.8` | Fraction of features sampled per tree. Reduces overfitting. |

### `models.ensemble`

| Key | Default | Description |
|---|---|---|
| `lstm_weight` | `0.4` | Weight given to LSTM probability outputs in the weighted average. |
| `xgboost_weight` | `0.6` | Weight given to XGBoost probability outputs. Must sum to 1.0 with `lstm_weight`. XGBoost gets higher weight by default because it tends to be more stable on tabular features. |

### `models` (top-level)

| Key | Default | Description |
|---|---|---|
| `model_mode` | `"lstm"` | Which model(s) to train and use for prediction. See options below. |
| `save_dir` | `data/models` | Root directory where trained model files are saved (`lstm.pt`, `xgboost.joblib`, `meta.joblib` per symbol). |
| `train_split` | `0.8` | Fraction of data used as the training split during hyperparameter tuning. The remaining 20% is the validation set. Final models are retrained on 100% of data. |
| `staleness_warning_days` | `30` | If a saved model is older than this many days, the system warns that it may be stale and should be retrained. |

**`model_mode` options:**

| Value | Description |
|---|---|
| `"lstm"` | Train and predict using the LSTM only. Faster training, good at capturing temporal patterns in the 60-timestep feature sequences. |
| `"xgboost"` | Train and predict using XGBoost only. Fast training, interpretable feature importances, no sequence context. |
| `"ensemble"` | Train both models and combine their probability outputs using per-stock dynamic weights derived from individual validation accuracies. |

> Changing `model_mode` requires retraining — existing saved models store the mode they were trained with and load correctly regardless of the current setting.

---

## 8. `signals`

Controls how model probability outputs are converted into trading signals.

```yaml
signals:
  confidence_threshold: 0.6
  strong_threshold: 0.8
  short_confidence_threshold: 0.7
  buy_return_threshold: 0.01
  sell_return_threshold: -0.01
```

| Key | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.6` | Minimum predicted probability for a BUY or SELL signal to be acted on. Below this the signal is downgraded to HOLD. |
| `strong_threshold` | `0.8` | Probability above which a signal is elevated to STRONG BUY or STRONG SELL. |
| `short_confidence_threshold` | `0.7` | Minimum confidence required to include a stock in the "Short" shortlist category. |
| `buy_return_threshold` | `0.01` | **Label threshold:** a day (or horizon period) with return ≥ +1% is labelled BUY during training. |
| `sell_return_threshold` | `-0.01` | **Label threshold:** a day (or horizon period) with return ≤ −1% is labelled SELL during training. Returns between the two thresholds are labelled HOLD. |

**Note:** `buy_return_threshold` and `sell_return_threshold` affect training labels — changing them requires retraining all models.

---

## 9. `screener`

Controls the stock screener that flags unusual volume or news activity.

```yaml
screener:
  volume_spike_threshold: 1.5
  news_volume_spike_threshold: 2.0
  news_lookback_days: 3
```

| Key | Default | Description |
|---|---|---|
| `volume_spike_threshold` | `1.5` | A stock is flagged as a volume spike if today's volume is ≥ 1.5× its recent average. |
| `news_volume_spike_threshold` | `2.0` | A stock is flagged as a news spike if its recent article count is ≥ 2.0× its baseline. |
| `news_lookback_days` | `3` | Number of recent days used to count news articles for the spike calculation. |

---

## 10. `report`

Controls how analysis reports are exported.

```yaml
report:
  export_dir: data/processed
  formats: ["console", "csv", "json"]
```

| Key | Default | Description |
|---|---|---|
| `export_dir` | `data/processed` | Directory where CSV and JSON report files are written. |
| `formats` | `["console", "csv", "json"]` | Output formats generated. Remove entries to suppress specific outputs (e.g. remove `"json"` to skip JSON export). |

---

## 11. `paper_trading`

Controls the paper trading ledger used by the `trade` and `portfolio` commands.

```yaml
paper_trading:
  ledger_file: data/trades/ledger.json
  report_dir: data/trades
```

| Key | Default | Description |
|---|---|---|
| `ledger_file` | `data/trades/ledger.json` | Path to the JSON file that records all paper trades. Delete this file to reset the paper portfolio. |
| `report_dir` | `data/trades` | Directory where gain/loss reports are exported. |

---

## Quick-reference: most commonly changed settings

| Goal | Setting | Suggested value |
|---|---|---|
| Predict next day instead of next week | `features.prediction_horizon` | `1` |
| Predict 2 weeks out | `features.prediction_horizon` | `10` |
| Switch LLM to Llama 3.1 | `llm.ollama.model` | `llama3.1:8b` |
| Reduce HOLD class bias (tighter band) | `signals.buy_return_threshold` / `sell_return_threshold` | `0.005` / `-0.005` |
| Speed up training (fewer trees) | `models.xgboost.n_estimators` | `200` |
| Reduce LSTM overfitting | `models.lstm.dropout` | `0.4` or `0.5` |
| Keep news fresher | `news.cache_expiry_hours` | `2` |
| Warn earlier about stale models | `models.staleness_warning_days` | `14` |
