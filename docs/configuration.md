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
    model: "qwen2.5:7b"
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
  prediction_horizon: 5
  use_financials: true
  financial_announcement_lag_days: 45
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
| `sequence_length` | `60` | Number of past trading days fed into the LSTM / Encoder-Decoder as one input sequence (~3 months). XGBoost and Prophet use a different representation. Increasing this requires proportionally more historical data. |

### `features.prediction_horizon`

| Key | Default | Description |
|---|---|---|
| `prediction_horizon` | `5` | How many trading days ahead the signal label is based on. The label is a **point-to-point return** (day `t` close vs day `t+horizon` close), not an average over the period. **Changing this requires retraining all models.** |

The horizon is selected per training run in the Streamlit UI (Train page → "Prediction horizon" selectbox) and is saved in each model's `meta.joblib`. At prediction time the horizon is loaded automatically from the saved model — the config value is only used as the default for the CLI.

**Supported horizon options (UI selectbox):**

| Value | Meaning | Buy/Sell threshold | Suitable for |
|---|---|---|---|
| `1` | Next trading day | ±1.0% | Short-term / intraday-style signals |
| `3` | 3 trading days | ±1.7% | Short swing trades |
| `5` | Same day next week ← **default** | ±2.2% | Weekly swing trading |
| `7` | ~1.5 calendar weeks | ±2.6% | Medium-term swing trading |
| `10` | ~2 calendar weeks | ±3.2% | Positional trades |

Thresholds widen with horizon following a √t volatility model. See [`signals.horizon_thresholds`](#signals-horizon_thresholds) to customise them.

### `features.use_financials`

| Key | Default | Description |
|---|---|---|
| `use_financials` | `true` | Include quarterly financial report features (margins, growth, leverage, ROE, cash flow, EPS surprise, and report aging). Requires internet access to fetch earnings data via yfinance. |

### `features.financial_announcement_lag_days`

| Key | Default | Description |
|---|---|---|
| `financial_announcement_lag_days` | `45` | Days after a fiscal quarter-end before the results are assumed to be public knowledge (used when the actual announcement date is unknown). Typical NIFTY 50 range: 30–60 days. A conservative 45-day default avoids look-ahead bias. |

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
  encoder_decoder:
    hidden_size: 128
    num_layers: 2
    dropout: 0.3
    learning_rate: 0.001
    epochs: 50
    batch_size: 32
    patience: 10
  prophet: {}
  ensemble:
    lstm_weight: 0.4
    xgboost_weight: 0.6
  selected_models: ["lstm"]
  save_dir: data/models
  train_split: 0.8
  staleness_warning_days: 30
```

### `models.lstm`

These are the **search space defaults** used by the hyperparameter tuner. The tuner tries combinations of hidden size, dropout, and learning rate and picks the best; the winner is then used for the full-data retrain.

| Key | Default | Description |
|---|---|---|
| `hidden_size` | `128` | Number of LSTM hidden units per layer. Tuner searches `[64, 128, 256]`. |
| `num_layers` | `2` | Number of stacked LSTM layers. Not tuned; change here to fix for all runs. |
| `dropout` | `0.3` | Dropout rate applied between LSTM layers and before the output head. Tuner searches `[0.2, 0.3]`. |
| `learning_rate` | `0.001` | Adam optimiser learning rate. Tuner uses `[0.001, 0.0005]`. |
| `epochs` | `50` | Maximum training epochs. Early stopping (via `patience`) usually stops before this. |
| `batch_size` | `32` | Mini-batch size for gradient updates. |
| `patience` | `10` | Early stopping patience — stops training if val loss does not improve for this many consecutive epochs. |

### `models.xgboost`

These are the **search space defaults** for the XGBoost tuner.

| Key | Default | Description |
|---|---|---|
| `n_estimators` | `500` | Maximum number of boosting trees. Actual count is determined by early stopping during tuning; the final full-data model scales this up by `1 / train_split`. |
| `max_depth` | `6` | Maximum tree depth. Tuner searches `[3, 4, 6, 8]`. Deeper trees capture more complex patterns but overfit more easily. |
| `learning_rate` | `0.05` | Shrinkage applied to each tree's contribution. Tuner searches `[0.05, 0.10]`. Lower values need more trees. |
| `early_stopping_rounds` | `20` | Stop boosting if val loss does not improve for this many rounds during tuning. Disabled (`None`) for the final full-data retrain. |
| `subsample` | `0.8` | Fraction of training rows sampled per tree. Reduces overfitting. |
| `colsample_bytree` | `0.8` | Fraction of features sampled per tree. Reduces overfitting. |

### `models.encoder_decoder`

Sequence-to-sequence LSTM that predicts a vector of `horizon` future price ratios (regression), then converts them to BUY/HOLD/SELL via a Gaussian CDF model.

| Key | Default | Description |
|---|---|---|
| `hidden_size` | `128` | Encoder and decoder LSTM hidden units. Tuner searches `[64, 128, 256]`. |
| `num_layers` | `2` | Stacked LSTM layers in both encoder and decoder. |
| `dropout` | `0.3` | Dropout rate. Tuner searches `[0.2, 0.3]`. |
| `learning_rate` | `0.001` | Adam learning rate. |
| `epochs` | `50` | Maximum epochs; early stopping applies. |
| `batch_size` | `32` | Mini-batch size. |
| `patience` | `10` | Early stopping patience. |

### `models.prophet`

```yaml
prophet: {}
```

Prophet has no architecture hyperparameters set here. Instead, `changepoint_prior_scale` and `seasonality_prior_scale` are grid-searched during tuning. The best values are found automatically and saved with the model.

### `models.ensemble`

When multiple models are selected, their probability outputs are combined via a **dynamic weighted average** derived from each model's individual validation balanced accuracy — better models automatically receive higher weight. The weights below are only used as a static fallback if dynamic weights cannot be computed.

| Key | Default | Description |
|---|---|---|
| `lstm_weight` | `0.4` | Fallback weight for LSTM in the weighted average. |
| `xgboost_weight` | `0.6` | Fallback weight for XGBoost. |

### `models` (top-level)

| Key | Default | Description |
|---|---|---|
| `selected_models` | `["lstm"]` | Default list of models to train. Overridden per run from the UI (multiselect) or CLI (`--models`). Multiple entries → ensemble. |
| `save_dir` | `data/models` | Root directory for trained model files. |
| `train_split` | `0.8` | Fraction of data used for training during hyperparameter search. The remaining 20% is the validation set. Final models are retrained on 100% of data. |
| `staleness_warning_days` | `30` | Warn in the Predict UI if a saved model is older than this many days. |

**`selected_models` options:**

| Value | Description |
|---|---|
| `["lstm"]` | LSTM sequence classifier. Captures temporal patterns across 60-day feature windows. |
| `["xgboost"]` | XGBoost tabular classifier. Fast training, interpretable feature importances, no sequence context. |
| `["encoder_decoder"]` | Encoder-Decoder LSTM regressor. Predicts multi-step price ratios; converts to BUY/HOLD/SELL. |
| `["prophet"]` | Facebook Prophet time-series model. Captures seasonality and trend; outputs a probabilistic signal. |
| `["lstm", "xgboost"]` | Two-model ensemble, dynamically weighted by validation accuracy. |
| `["lstm", "xgboost", "encoder_decoder", "prophet"]` | Full ensemble — all four models combined. |

> Each saved model records the models it was trained with and the features used. Prediction always matches training conditions regardless of the current config value.

**What is saved in `meta.joblib` per symbol:**

| Field | Description |
|---|---|
| `scaler` | Fitted `StandardScaler` for tabular features |
| `seq_scaler` | Fitted `StandardScaler` for sequence features |
| `feature_names` | Ordered list of feature column names used during training |
| `input_size` | Feature dimension |
| `lstm_weight` / `xgb_weight` / `ed_weight` / `prophet_weight` | Dynamic ensemble weights derived from validation accuracy |
| `selected_models` | List of model IDs that were trained |
| `trained_at` | ISO-format timestamp of when training completed |
| `horizon` | Prediction horizon (days) used during training |
| `use_news` | Whether news features were enabled during training |
| `use_llm` | Whether LLM features were enabled during training |
| `use_financials` | Whether financial report features were enabled during training |
| `val_accuracy` | Weighted validation balanced accuracy at training time |

---

## 8. `signals`

Controls how model probability outputs are converted into trading signals, and defines the BUY/SELL/HOLD label thresholds used during training.

```yaml
signals:
  confidence_threshold: 0.6
  strong_threshold: 0.8
  short_confidence_threshold: 0.7
  buy_return_threshold: 0.01
  sell_return_threshold: -0.01
  horizon_thresholds:
    1:  [ 0.010, -0.010]
    2:  [ 0.014, -0.014]
    3:  [ 0.017, -0.017]
    4:  [ 0.020, -0.020]
    5:  [ 0.022, -0.022]
    6:  [ 0.024, -0.024]
    7:  [ 0.026, -0.026]
    8:  [ 0.028, -0.028]
    9:  [ 0.030, -0.030]
    10: [ 0.032, -0.032]
```

### Confidence thresholds

| Key | Default | Description |
|---|---|---|
| `confidence_threshold` | `0.6` | Minimum predicted probability for a BUY or SELL signal to be acted on. Below this the signal is downgraded to HOLD. |
| `strong_threshold` | `0.8` | Probability above which a signal is elevated to STRONG BUY or STRONG SELL. |
| `short_confidence_threshold` | `0.7` | Minimum confidence required to include a stock in the short-selling candidate list. |

### `signals.horizon_thresholds` (label thresholds)

These are the **training label thresholds** — they determine what counts as a BUY or SELL during dataset creation. Each row is `horizon_days: [buy_threshold, sell_threshold]`.

| Horizon | BUY if `horizon_return ≥` | SELL if `horizon_return ≤` | Else |
|---|---|---|---|
| 1 day | +1.0% | −1.0% | HOLD |
| 3 days | +1.7% | −1.7% | HOLD |
| 5 days | +2.2% | −2.2% | HOLD |
| 7 days | +2.6% | −2.6% | HOLD |
| 10 days | +3.2% | −3.2% | HOLD |

**Design:** thresholds scale as √horizon from a ±1% base at horizon=1, matching how volatility compounds under a random walk assumption. Wider bands at longer horizons prevent the HOLD class from collapsing.

**Editing:** you can widen or narrow the HOLD band by changing any entry here without touching code. **Changing thresholds requires retraining** — the labels embedded in the training data will differ.

### `signals.buy_return_threshold` / `sell_return_threshold`

| Key | Default | Description |
|---|---|---|
| `buy_return_threshold` | `0.01` | Fallback label threshold used only if `horizon_thresholds` is missing from the config. |
| `sell_return_threshold` | `-0.01` | Fallback label threshold (same condition). |

> In normal use these fallbacks are never reached because `horizon_thresholds` covers all supported horizons (1–10).

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
| Default to next-day predictions | `features.prediction_horizon` | `1` |
| Default to 2-week predictions | `features.prediction_horizon` | `10` |
| Widen HOLD band (fewer BUY/SELL signals) | `signals.horizon_thresholds` | Increase absolute values, e.g. `5: [0.030, -0.030]` |
| Narrow HOLD band (more signals) | `signals.horizon_thresholds` | Decrease absolute values, e.g. `5: [0.015, -0.015]` |
| Switch LLM to Llama 3.1 | `llm.ollama.model` | `llama3.1:8b` |
| Speed up training (fewer trees) | `models.xgboost.n_estimators` | `200` |
| Reduce LSTM overfitting | `models.lstm.dropout` | `0.4` or `0.5` |
| Keep news fresher | `news.cache_expiry_hours` | `2` |
| Warn earlier about stale models | `models.staleness_warning_days` | `14` |
| Disable financial features globally | `features.use_financials` | `false` |
| Extend financial data lag (conservative) | `features.financial_announcement_lag_days` | `60` |
