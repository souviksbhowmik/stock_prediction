# Technical Design Document

## Indian Stock Market Prediction System

---

## 1. System Architecture Overview

The system is a modular Python application for ML-powered stock prediction on Indian markets (NSE/BSE). It combines technical analysis, news sentiment, and LLM-based broker insights into an ensemble ML pipeline, with paper trading simulation and stock screening capabilities.

### Module Map

```
src/stock_prediction/
├── cli.py                  # Click-based CLI (13 commands)
├── config.py               # YAML config loader with caching
├── data/                   # Data providers (yfinance)
│   ├── base.py             # Abstract DataProvider, StockData dataclass
│   └── yfinance_provider.py
├── features/               # Feature engineering
│   ├── technical.py        # 22+ technical indicators
│   ├── news_based.py       # News/LLM feature merging
│   └── pipeline.py         # Orchestrates feature construction + labeling
├── models/                 # ML models
│   ├── lstm_model.py           # StockLSTM (PyTorch classifier)
│   ├── xgboost_model.py        # XGBoostPredictor (tabular classifier)
│   ├── encoder_decoder_model.py # Seq2seq LSTM regressor → signal
│   ├── prophet_model.py        # Facebook Prophet regressor → signal
│   ├── tft_model.py            # Temporal Fusion Transformer regressor → signal
│   ├── qlearning_model.py      # Tabular Q-learning RL agent
│   ├── dqn_model.py            # Deep Q-Network RL agent
│   ├── ensemble.py             # Weighted ensemble of all model types
│   └── trainer.py              # Training orchestration + model persistence
├── signals/                # Signal generation + screening
│   ├── generator.py        # TradingSignal, confidence thresholds
│   ├── screener.py         # StockScreener (suggest, shortlist, screen)
│   ├── paper_trading.py    # Paper trade lifecycle + ledger
│   └── report.py           # Rich console display + CSV/JSON export
├── news/                   # News & NLP pipeline
│   ├── rss_fetcher.py      # Google News RSS fetcher
│   ├── sentiment.py        # FinBERT sentiment analyzer
│   ├── ner.py              # spaCy NER + entity linking
│   └── news_features.py    # Windowed sentiment aggregation
├── llm/                    # LLM integration
│   ├── base.py             # Abstract LLMProvider
│   ├── ollama_provider.py  # Ollama local LLM
│   └── news_analyzer.py    # BrokerNewsAnalyzer (structured prompts)
└── utils/
    ├── constants.py         # NIFTY 50 tickers, aliases, sector map
    └── logging.py           # Logger setup
```

### Entry Point

- **Package:** `stock_prediction.cli:cli`
- **Executable:** `stockpredict` (installed via pip/pyproject.toml)
- **Framework:** Click with `@click.group()` and 13 subcommands

---

## 2. Data Layer

### DataProvider (Abstract Base)

**File:** `src/stock_prediction/data/base.py`

```
class DataProvider(ABC):
    fetch_historical(symbol, start_date, end_date) -> StockData
    fetch_batch(symbols, start_date, end_date) -> list[StockData]
    fetch_latest(symbol) -> StockData
```

**StockData** is a `@dataclass` container holding:
- `symbol: str` — ticker (e.g., `RELIANCE.NS`)
- `df: DataFrame` — OHLCV data
- `metadata: dict` — company name, sector, market cap

### YFinanceProvider

**File:** `src/stock_prediction/data/yfinance_provider.py`

- Fetches OHLCV data via the `yfinance` library
- Column standardization: converts to Title case (`Open`, `High`, `Low`, `Close`, `Volume`)
- Metadata extraction from `ticker.info` (company name, sector, market cap)
- Default start date: `2020-01-01` (from `settings.yaml`)

### Factory

```python
# data/__init__.py
get_provider(name="yfinance") -> DataProvider
```

---

## 3. Feature Engineering Pipeline

**File:** `src/stock_prediction/features/pipeline.py`

### Class: FeaturePipeline

Orchestrates all feature construction. Initialized with `use_news: bool` and `use_llm: bool` flags.

#### build_features(symbol, start_date, end_date) -> DataFrame

Pipeline steps:
1. Fetch OHLCV data via DataProvider
2. Add technical indicators (`add_technical_indicators`)
3. Merge news features (if `use_news=True`)
4. Merge LLM features (if `use_llm=True`)
5. Add labels (`return_1d`, `return_5d`, `signal`)
6. Drop NaN rows

#### prepare_training_data(symbol, start_date, end_date)

Returns `(sequences, tabular, labels, feature_names)`:
- **Sequences:** Shape `(N, 60, n_features)` — 60-day rolling windows for LSTM
- **Tabular:** Shape `(N, n_features)` — last row of each window for XGBoost
- **Labels:** `signal` column mapped from 1-day returns

Raises `ValueError` (instead of returning empty arrays) in two cases:
- **No price data** — yfinance returned nothing for the symbol (wrong ticker, delisted, network issue)
- **Insufficient rows** — fewer than `sequence_length + 10` (70) rows remain after `dropna()`; requires ~130+ raw trading days. Error message includes the actual row count and a hint to use an earlier `--start-date`.

### Label Creation

| Condition | Signal | Value |
|-----------|--------|-------|
| `return_1d >= 0.01` | BUY | 2 |
| `return_1d <= -0.01` | SELL | 0 |
| Otherwise | HOLD | 1 |

Thresholds are configurable via `signals.buy_return_threshold` and `signals.sell_return_threshold`.

### Technical Indicators

**File:** `src/stock_prediction/features/technical.py`

Function `add_technical_indicators(df)` adds the following using the `ta` library. Requires minimum 30 rows.

| Category | Indicators | Parameters |
|----------|-----------|------------|
| Momentum | RSI | period=14 |
| Trend | MACD, MACD_Signal, MACD_Hist | fast=12, slow=26, signal=9 |
| Trend | SMA_20, SMA_50 | periods=[20, 50] |
| Trend | EMA_12, EMA_26 | periods=[12, 26] |
| Volatility | BB_Upper, BB_Middle, BB_Lower, BB_Width | period=20, std=2 |
| Volatility | ATR | period=14 |
| Volume | OBV | — |
| Volume | VWAP | approximation |
| Oscillator | Stoch_K, Stoch_D | period=14 |
| Derived | SMA_20_50_Cross | bullish crossover flag |
| Derived | Price_SMA20_Ratio, Price_SMA50_Ratio | price/SMA ratios |
| Derived | Volume_SMA20, Volume_Ratio | 20-day volume average & ratio |

### News Feature Merging

**File:** `src/stock_prediction/features/news_based.py`

- `merge_news_features(price_df, news_features_dict)` — adds columns prefixed `news_*`
- `merge_llm_features(price_df, llm_scores_dict)` — adds columns prefixed `llm_*`

---

## 4. News and Sentiment Layer

### RSS Fetcher

**File:** `src/stock_prediction/news/rss_fetcher.py`

**Class: GoogleNewsRSSFetcher**

- **Source:** Google News RSS (`https://news.google.com/rss/search`)
- **Default queries:** "Indian stock market", "NSE BSE stocks", "NIFTY 50", "Indian economy"
- **Max articles:** 50 per query
- **Deduplication:** MD5 hash of URL
- **Rate limiting:** 1-second delay between requests
- **Cache:** JSON files in `data/news_cache/`, keyed by MD5 of query, 6-hour expiry

**NewsArticle dataclass:**
- Fields: `title`, `source`, `published` (datetime), `url`, `snippet`, `query`, `article_id`
- Methods: `to_dict()`, `from_dict()`

### Sentiment Analysis (FinBERT)

**File:** `src/stock_prediction/news/sentiment.py`

**Class: FinancialSentimentAnalyzer**

- **Model:** `ProsusAI/finbert` (DistilBERT fine-tuned on financial text)
- **Output:** 3-class probabilities (positive, negative, neutral)
- **Processing:** Batch size 32, truncate to 512 tokens, `torch.no_grad()`
- **Score:** `positive_score - negative_score` (range: -1 to +1)

**SentimentResult dataclass:**
- `label`, `score`, `positive_score`, `negative_score`, `neutral_score`

### Named Entity Recognition & Entity Linking

**File:** `src/stock_prediction/news/ner.py`

**Class: StockEntityLinker**

- **Model:** spaCy `en_core_web_sm`
- **Entity types:** ORG, PERSON, GPE, PRODUCT
- **3-layer matching:**
  1. Direct alias matching (case-insensitive) against `COMPANY_ALIASES`
  2. NER entity lookup against aliases
  3. Bare ticker symbol search (e.g., "TCS", "INFY")
- **Result:** List of matched NIFTY 50 tickers

### News Feature Generation

**File:** `src/stock_prediction/news/news_features.py`

**Class: NewsFeatureGenerator**

Generates windowed features across **1-day, 7-day, and 30-day** windows:

| Feature | Description |
|---------|-------------|
| `sentiment_{window}_mean` | Average sentiment score |
| `sentiment_{window}_std` | Sentiment volatility |
| `news_volume_{window}` | Article count |
| `positive_ratio_{window}` | % positive articles |
| `negative_ratio_{window}` | % negative articles |
| `sentiment_{window}_trend` | Slope (7d, 30d only) |
| `{category}_{window}` | Keyword category counts |

**Keyword categories** (from `constants.py`): `earnings`, `merger`, `regulation`, `management`, `dividend`, `expansion` — ~50 total keyword terms.

---

## 5. LLM Integration

### Abstract Provider

**File:** `src/stock_prediction/llm/base.py`

```python
class LLMProvider(ABC):
    analyze(prompt: str) -> str
    analyze_batch(prompts: list[str]) -> list[str]
    is_available() -> bool
```

### OllamaProvider

**File:** `src/stock_prediction/llm/ollama_provider.py`

- **Model:** `llama3.1:8b` (configurable)
- **Base URL:** `http://localhost:11434`
- **Timeout:** 120 seconds
- **Health check:** Attempts to list models via Ollama API
- **Error handling:** Returns empty string on failure

### BrokerNewsAnalyzer

**File:** `src/stock_prediction/llm/news_analyzer.py`

Sends top 15 news headlines to the LLM with a structured prompt (`BROKER_ANALYSIS_PROMPT`) requesting a JSON response with 10 score dimensions (0-10 scale):

| Score Dimension | Description |
|-----------------|-------------|
| `earnings_outlook` | Earnings trajectory |
| `competitive_position` | Market position strength |
| `management_quality` | Leadership quality |
| `sector_momentum` | Sector-level trends |
| `risk_level` | Risk assessment |
| `growth_catalyst` | Growth potential catalysts |
| `valuation_signal` | Valuation attractiveness |
| `institutional_interest` | Institutional buying signals |
| `macro_impact` | Macroeconomic sensitivity |
| `overall_broker_score` | Composite score |

Plus a 1-2 sentence `_summary`.

- **Cache:** 24-hour expiry in `data/news_cache/`, keyed by MD5(symbol + date)
- **Fallback:** Returns neutral scores (all 5.0) if LLM is unavailable
- **Response parsing:** Extracts JSON from response text, clamps scores to 0-10

### Factory

```python
# llm/__init__.py
get_llm_provider(name="ollama") -> LLMProvider
```

---

## 6. ML Models

### LSTM Architecture

**File:** `src/stock_prediction/models/lstm_model.py`

**Class: StockLSTM(nn.Module)**

```
Input: (batch, seq_len=60, n_features)
  ↓
LSTM(input_size=n_features, hidden_size=128, num_layers=2, dropout=0.3)
  ↓  (extract last hidden state)
Dense(128 → 64) + ReLU + Dropout(0.3)
  ↓
Dense(64 → 3)  →  Softmax  →  [P(SELL), P(HOLD), P(BUY)]
```

**Class: LSTMPredictor** (training wrapper)

| Hyperparameter | Value |
|----------------|-------|
| hidden_size | 128 |
| num_layers | 2 |
| dropout | 0.3 |
| learning_rate | 0.001 |
| epochs | 50 (max) |
| batch_size | 32 |
| patience | 10 (early stopping) |
| loss | CrossEntropyLoss |
| optimizer | Adam |

- Device auto-detection: MPS (Apple Silicon) → CUDA → CPU
- Saves as `.pt` with `state_dict` + metadata

### XGBoost Configuration

**File:** `src/stock_prediction/models/xgboost_model.py`

**Class: XGBoostPredictor**

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| eval_metric | mlogloss |
| early_stopping_rounds | 20 |

- Input: Tabular features `(N, n_features)`
- Output: Probabilities `(N, 3)`
- Feature importance via `get_feature_importance()`
- Saves as `.joblib` with model + feature_names

### Encoder-Decoder Architecture

**File:** `src/stock_prediction/models/encoder_decoder_model.py`

Sequence-to-sequence LSTM regressor. Predicts a vector of `horizon` future
price ratios rather than a direct class label, then converts those ratios to
BUY / HOLD / SELL probabilities using a Gaussian CDF model.

**Architecture:**
```
Input: (batch, seq_len=60, n_features)
  ↓
Encoder LSTM (hidden_size, num_layers, dropout)
  → (h_n, c_n)  [final hidden and cell state]
  ↓
Decoder LSTM — one step per forecast horizon
  input at step k: [last_encoder_output | prev_pred_ratio]  (teacher-forcing during training)
  → outputs ratio_k = close[t+k] / close[t]
  ↓
Output: (batch, horizon)  price ratios
```

**Regression → probability conversion:**

1. Compute per-step residuals on the validation set:
   `residual_k = predicted_ratio_k − true_ratio_k`
2. Fit a single Gaussian: `μ=0`, `σ = std(all residuals across steps)`
3. At inference, for horizon step `k`, compute:
   - `P(BUY)  = 1 − Φ((buy_threshold  − predicted_ratio_k) / σ)`
   - `P(SELL) = Φ((sell_threshold − predicted_ratio_k) / σ)`
   - `P(HOLD) = 1 − P(BUY) − P(SELL)`
   where `Φ` is the standard normal CDF and thresholds come from
   `signals.horizon_thresholds[horizon]`.
4. Softmax-normalise the three values so they sum to 1.0.

| Property | Value |
|---|---|
| Loss | MSE on price ratios |
| Tuning metric | MAPE on validation set (lower → better) |
| Teacher forcing ratio | 0.5 during tuning, 0.3 during full retrain |
| Saves as | `.pt` (PyTorch state_dict + residual_std + horizon) |

---

### Temporal Fusion Transformer (TFT)

**File:** `src/stock_prediction/models/tft_model.py`

Pure-PyTorch implementation of a simplified TFT using gated residual networks,
variable selection, LSTM temporal encoding, and multi-head self-attention.
Uses the same regression → Gaussian CDF probability pipeline as the
Encoder-Decoder, so output probabilities are directly comparable.

**Building blocks:**

#### GatedResidualNetwork (GRN)

```
x  ──► Linear(d→d) ──► ELU ──► Linear(d→d) ──► GLU gate
x  ──────────────────────────────────────────► skip (Linear if dim mismatch)
                                                  ↓
                                             LayerNorm(gate + skip)
```

The GLU (Gated Linear Unit) gate applies sigmoid to half the channels and
multiplies element-wise with the other half, giving the network a soft
content-selection mechanism. The skip connection with LayerNorm stabilises
training in deeper configurations.

#### VariableSelectionNetwork (VSN)

```
Input: (batch, seq_len, n_features)  — each feature is a scalar
  ↓  for each feature j:
  GRN_j(x[:,j])                         → (batch, seq_len, hidden_size)
  ↓  all features stacked:
  GRN_attn(flatten(x))                  → (batch, seq_len, n_features)
  softmax(·) over feature axis           → attention weights ω_j
  ↓
  weighted_sum(ω_j * GRN_j outputs)     → (batch, seq_len, hidden_size)
```

Attention weights ω indicate which features the model relies on at each
timestep. They are inspectable post-training for feature importance analysis.

#### Full TFT forward pass

```
Input: (batch, seq_len, n_features)
  ↓
VariableSelectionNetwork               → (batch, seq_len, hidden_size)
  ↓
LSTM encoder (num_lstm_layers)         → (batch, seq_len, hidden_size)
  ↓
Multi-head Self-Attention
  Q = K = V = LSTM output
  num_heads, head_dim = hidden_size // num_heads
  → (batch, seq_len, hidden_size)
  ↓
GatedResidualNetwork (post-attention)  → (batch, seq_len, hidden_size)
  ↓
Mean-pool over seq_len                 → (batch, hidden_size)
  ↓
Linear(hidden_size → horizon)          → (batch, horizon)  price ratios
```

`num_heads` is auto-reduced to 1 if `hidden_size % num_heads ≠ 0`.

| Property | Value |
|---|---|
| Loss | MSE on price ratios |
| Tuning metric | MAPE (same as Encoder-Decoder) |
| Probability | Gaussian CDF → softmax (same pipeline as Encoder-Decoder) |
| Saves as | `.pt` (PyTorch state_dict + residual_std + horizon) |

---

### Reinforcement Learning — Shared Reward Function

Both Q-learning and DQN use the **position-based differential P&L reward**
from Moody & Saffell (2001) and Spooner et al. (2018). The agent manages a
binary position: `1 = LONG`, `0 = FLAT`.

#### Formal definition

```
reward(t) = new_position(t) × r(t)  −  |Δposition(t)| × tc
```

| Symbol | Definition |
|---|---|
| `new_position(t)` | Position *after* applying action `a(t)`: 1 (LONG) or 0 (FLAT) |
| `r(t)` | 1-step return at bar `t` = `close[t+1] / close[t] − 1` |
| `Δposition(t)` | `\|new_position(t) − old_position(t)\|` ∈ {0, 1} — 1 when a trade is opened or closed |
| `tc` | Transaction cost fraction (default 0.001 = 0.1% per trade side) |

#### Case-by-case breakdown

| Action | Prior position | new\_position | Δposition | Reward formula | Economic meaning |
|--------|---------------|--------------|-----------|----------------|-----------------|
| **BUY** | FLAT (0) | 1 | 1 | `r(t) − tc` | Open a long position; earn return for the bar; pay entry cost |
| **BUY** | LONG (1) | 1 | 0 | `r(t)` | Already long; no trade, no cost; earn the bar's return |
| **SELL** | LONG (1) | 0 | 1 | `0 − tc = −tc` | Close long position; pay exit cost; return for bar goes to seller |
| **SELL** | FLAT (0) | 0 | 0 | `0` | Already flat; no trade, no cost, no P&L |
| **HOLD** | LONG (1) | 1 | 0 | `r(t)` | Keep position; earn unrealised gain/loss |
| **HOLD** | FLAT (0) | 0 | 0 | `0` | Stay flat; zero exposure, zero cost |

#### Design properties

- **P&L accumulation** — the agent earns `r(t)` for every bar it holds a long
  position, matching real equity-curve accounting.
- **Cost on transition only** — `tc` is charged exactly once when opening
  (`FLAT → LONG`) or closing (`LONG → FLAT`); repeated same-direction actions
  incur no extra cost, which discourages churning.
- **No short selling** — the agent cannot go below `position = 0`; a SELL
  action from a flat position is a no-op (`reward = 0`).
- **Horizon mismatch by design** — the reward is 1-step P&L, not the
  horizon-based label used to evaluate classification accuracy. The agent is
  optimised as a trading policy, and accuracy against horizon labels is an
  *external* evaluation metric reported separately.

#### Label source for reward

`r(t)` is taken from the first column of `reg_targets` produced by
`prepare_regression_data`:

```python
returns_1d = reg_targets[:, 0] - 1.0   # close[t+1]/close[t] − 1
```

This is the *same* data used by the Encoder-Decoder and TFT for regression
targets, so no additional data pipeline is needed for the RL agents.

---

### Tabular Q-Learning

**File:** `src/stock_prediction/models/qlearning_model.py`

Tabular reinforcement learning agent. The Q-function is stored as an explicit
dictionary mapping discrete state tuples to Q-value arrays. No neural
network is involved.

#### State space design

```
Configured state features (default):
  RSI, MACD_Histogram, BB_Width, Volume_Ratio, Price_Momentum_5d

For each feature f:
  1. Fit n_bins quantile edges on the training column  (n_bins = 3 → 4 edges)
  2. At inference: bin_idx = searchsorted(inner_edges, feature_value)
                             ∈ {0, 1, ..., n_bins−1}

State = tuple(bin_idx_1, bin_idx_2, ..., bin_idx_k)
```

With 5 features and 3 bins each: **3^5 = 243 possible states**.
Unseen states at inference receive a HOLD-biased prior
`Q = [0.0, 0.01, 0.0]` (SELL / HOLD / BUY), so the agent defaults to doing
nothing rather than random trading when it encounters an unknown market regime.

#### Q-learning update rule

```
Q(s, a) ← Q(s, a) + α × [r + γ × max_{a'} Q(s', a') − Q(s, a)]
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| `α` (learning_rate) | Step size; how much each new experience overrides the old estimate | 0.1 |
| `γ` (discount_factor) | Importance of future rewards vs immediate reward | 0.95 |
| `r` | Reward from the shared reward function | — |
| `max_{a'} Q(s', a')` | Best Q-value in next state (greedy target) | — |

#### Exploration schedule

```
ε_episode = max(ε_end, ε_start × ε_decay^episode)
```

Epsilon decays per episode (not per step), so the agent explores the full
time-series multiple times at each exploration level before reducing ε.
Default: ε_start=1.0, ε_decay=0.995, ε_end=0.01.

#### Probability output

```
Q_values = Q_table.get(state, UNSEEN_PRIOR)          # shape (3,)
proba = softmax(Q_values / temperature)               # shape (3,)
```

Lower temperature → more peaked distribution (more decisive signals).
Higher temperature → flatter distribution (more uncertainty).

#### Input

Uses the **last timestep** of the scaled sequence window:
`X_seq[:, -1, :]` (shape `(N, n_features)`) — the most recent bar's
feature vector, which represents the agent's current observation of the market.

| Property | Value |
|---|---|
| Saves as | `.joblib` (Q-table dict + bin edges + state feature indices + hyperparams) |
| Tuning grid | 5 configs varying n_bins ∈ {3,4,5}, α ∈ {0.05,0.10}, n_episodes ∈ {10,20} |

---

### Deep Q-Network (DQN)

**File:** `src/stock_prediction/models/dqn_model.py`

Neural reinforcement learning agent. Uses the **same reward function** as
tabular Q-learning but replaces the discrete Q-table with a continuous neural
Q-function, adding experience replay and a target network for training
stability.

#### Comparison with tabular Q-learning

| Property | Tabular Q-Learning | Deep Q-Network |
|---|---|---|
| State representation | Discretised bins | Raw continuous features |
| Q-function | Lookup table | MLP neural network |
| Generalisation | None (unseen states use prior) | Interpolates via learned weights |
| Update | Single-sample TD | Mini-batch from replay buffer |
| Training stability | Inherently stable (table lookup) | Target network + gradient clipping |
| State space | 3^5 = 243 cells (default) | Continuous ℝ^n |

#### Network architecture

```
_QNetwork (MLP):

Input: (batch, n_features)   — continuous scaled feature vector
  ↓
[Linear(n_features → h_i) → ReLU → Dropout(p)]  ×  len(hidden_sizes)
  ↓
Linear(h_last → 3)           — one Q-value per action (SELL / HOLD / BUY)

Output: (batch, 3)           — unnormalised Q-values
```

Default `hidden_sizes = [256, 128]` gives a two-hidden-layer network with
256 → 128 → 3 units. Adding entries (e.g. `[256, 128, 64]`) creates a
deeper network.

#### Experience replay buffer

```
_ReplayBuffer (circular deque, capacity = buffer_size):

  push(state, action, reward, next_state, done)   → appends; oldest evicted when full
  sample(batch_size)                               → random minibatch of transitions
```

Random sampling breaks the temporal autocorrelation of consecutive market
bars, which would otherwise bias gradient updates toward recent market
conditions. The buffer accumulates all transitions from all episodes.

#### Bellman update

```
For each minibatch transition (s, a, r, s', done):

  q_pred   = Q_online(s)[a]                               current Q-value for taken action
  q_next   = max_a' Q_target(s')                          best next Q-value (from frozen target net)
  q_target = r + γ × q_next × (1 − done)                 Bellman target

  loss = Huber(q_pred, q_target)                          smooth-L1; robust to reward outliers
  ∇ backprop through Q_online only
  clip gradients: max_norm = 10.0
```

`Q_target` is a **frozen copy** of `Q_online` synced every `target_update_freq`
gradient steps. Without the target network, both sides of the Bellman equation
change simultaneously, causing training instability (the moving-target problem).

#### Linear epsilon schedule

```
total_train_steps = (N − 1) × n_episodes

At global step k:
  ε(k) = max(ε_end, ε_start − (ε_start − ε_end) × k / total_train_steps)
```

Epsilon decays **linearly per environment step** from `ε_start=1.0` to
`ε_end=0.01` over the full training run. This guarantees:
- Early training: high ε → diverse random transitions fill the buffer
- Late training: low ε → mostly greedy actions based on learned Q-values

This is deliberately different from the tabular agent's per-episode exponential
decay, because neural networks require many gradient steps to propagate
Q-value information across the weight space — the agent must actually exploit
its learned policy to generate useful on-policy data for further updates.

| Property | Value |
|---|---|
| Loss | Huber / smooth-L1 |
| Gradient clipping | `max_norm = 10.0` |
| Optimizer | Adam |
| Saves as | `.pt` (online network state_dict + all hyperparams) |
| Tuning grid | 5 configs varying hidden_sizes ∈ {[128,64],[256,128],[128,64,32]}, dropout ∈ {0.2,0.3}, lr ∈ {0.001,0.0005} |

---

### Ensemble Weighting

**File:** `src/stock_prediction/models/ensemble.py`

**Class: EnsembleModel** — supports up to seven model types simultaneously.

```
ensemble_probs = Σ (weight_i × model_i.predict_proba()) / Σ weight_i
signal = argmax(ensemble_probs)   # 0=SELL, 1=HOLD, 2=BUY
confidence = max(ensemble_probs)
```

Weights are derived dynamically from each model's validation balanced accuracy (normalised so they sum to 1.0). A model's weight is proportional to how well it performed on the held-out validation set — better models automatically dominate the ensemble.

**EnsemblePrediction dataclass fields:**
- `signal`, `signal_idx`, `confidence`, `probabilities`
- `lstm_probs`, `xgboost_probs`, `encoder_decoder_probs`, `prophet_probs`
- `tft_probs`, `qlearning_probs`, `dqn_probs`

### Model Trainer

**File:** `src/stock_prediction/models/trainer.py`

**Class: ModelTrainer**

**`train_batch` return value:** `dict[str, dict]` — one entry per symbol with keys:

| Key | Type | Description |
|-----|------|-------------|
| `status` | str | `'success'`, `'no_data'`, or `'failed'` |
| `reason` | str | Specific failure message; empty string on success |
| `accuracy` | float\|None | Validation accuracy; `None` on failure |
| `model` | EnsembleModel\|None | Trained model; `None` on failure |

`ValueError` from the feature pipeline is caught as `no_data`; all other exceptions are caught as `failed`. After every run, results are written to `data/reports/train_summary.csv`.

**`train_stock` return value:** `(EnsembleModel | None, float | None)` — model and validation accuracy.

Training workflow:
1. Feature pipeline builds features with labels — raises `ValueError` if data is missing or insufficient
2. Chronological train/val split (80/20, no shuffling)
3. `StandardScaler` fitted on training data for both sequences and tabular
4. LSTM training with early stopping on validation loss
5. XGBoost training with early stopping on validation mlogloss
6. Ensemble validation accuracy computed on held-out val set
7. Save all artifacts
8. Write `data/reports/train_summary.csv` with per-symbol status, accuracy, and failure reason

**Model loading:**
- `load_models(symbol)` returns `(ensemble, scaler, seq_scaler, model_age_days)`
- Staleness warning if model age > 30 days (configurable)

---

## 7. Signal Generation

**File:** `src/stock_prediction/signals/generator.py`

### TradingSignal Dataclass

Fields: `symbol`, `signal`, `confidence`, `strength`, `short_score`, `is_short_candidate`, `probabilities`, `weekly_outlook`, `monthly_outlook`, `llm_summary`, `top_headlines`, `technical_summary`

### Signal Logic

**Class: SignalGenerator**

| Condition | Output Signal |
|-----------|---------------|
| `confidence < 0.6` | HOLD |
| Base=BUY, `confidence >= 0.8` | STRONG BUY |
| Base=BUY, `confidence < 0.8` | BUY |
| Base=SELL, `confidence >= 0.8` | STRONG SELL |
| Base=SELL, `confidence < 0.8` | SELL |
| Base=HOLD | HOLD |

### Short Score Computation

`_compute_short_score(prediction, technical_data) -> float (0-1)`

| Component | Weight | Logic |
|-----------|--------|-------|
| Sell probability | 40% | Direct from ensemble P(SELL) |
| RSI overbought | 20% | If RSI > 70: scaled by `(RSI-70)/30` |
| MACD bearish | 20% | If MACD histogram < 0 |
| Below SMA50 | 20% | If Price/SMA50 ratio < 1.0 |

**Short candidate criteria:** `signal=SELL AND confidence >= 0.7 AND short_score >= 0.5`

---

## 8. Stock Screening and Shortlisting

**File:** `src/stock_prediction/signals/screener.py`

### StockScreener Class

#### suggest(count=10)

Scores each NIFTY 50 stock:

| Component | Points | Condition |
|-----------|--------|-----------|
| Momentum | 0-3 (normalized) | `1w_return * 0.6 + 1m_return * 0.4` |
| Volume spike | +2 | `Volume_Ratio > 1.5` |
| RSI oversold | +1.5 | `RSI < 30` |
| RSI overbought | +1.0 | `RSI > 70` |
| SMA crossover | +2 | Bullish 20/50 crossover |
| 52-week high | +1.5 | `price >= 95% of 52w high` |
| News mentions | +0.5 each (max 2.0) | Article mention count |

Sorted by total score descending.

#### shortlist(count=5)

Three output categories:
- **Buy candidates:** Top N NIFTY 50 by composite score
- **Short candidates:** Bottom N by short score (negative momentum, overbought RSI, below SMA50, bearish crossover)
- **Trending:** Non-NIFTY stocks discovered from news (mention count >= 1)

#### screen(symbols)

Full screening with four sections:
1. **Pre-screened top picks** — volume spikes + technical signals
2. **Sector momentum** — stocks ranked within their sectors
3. **News alerts** — breaking news discoveries
4. **Full rankings** — all symbols ranked

---

## 9. Paper Trading

**File:** `src/stock_prediction/signals/paper_trading.py`

### Trade Lifecycle

```
BUY (amount)  →  OPEN LONG  →  SELL  →  CLOSED (PnL = (exit - entry) × qty)
SHORT (amount) → OPEN SHORT → COVER →  CLOSED (PnL = (entry - exit) × qty)
```

### PaperTrade Dataclass

- `trade_id`: 8-character UUID
- `symbol`, `trade_type` (LONG/SHORT), `entry_date`, `entry_price`, `quantity`, `amount`
- `exit_date`, `exit_price`, `status` (OPEN/CLOSED), `pnl`, `pnl_pct`

### PaperTradingManager

| Method | Action |
|--------|--------|
| `buy(symbol, amount)` | Open LONG or cover SHORT |
| `sell(symbol, trade_id?)` | Close LONG or SHORT position |
| `short_sell(symbol, amount)` | Open SHORT position |
| `cover_short(symbol, trade_id?)` | Close SHORT position |
| `get_portfolio()` | All OPEN trades + unrealized PnL |
| `calculate_gains()` | GainReport for all CLOSED trades |

### Ledger Persistence

- **File:** `data/trades/ledger.json`
- **Format:** JSON array of serialized PaperTrade dicts
- Read/write on every operation (no in-memory caching)

### GainReport Dataclass

- `total_trades`, `winning_trades`, `losing_trades`
- `total_pnl` (INR), `total_pnl_pct`
- `best_trade`, `worst_trade` (with details)
- `per_stock` breakdown: {symbol: {trades, pnl, pnl_pct}}
- `open_positions`, `unrealized_pnl`

---

## 10. Reporting

**File:** `src/stock_prediction/signals/report.py`

### ReportFormatter Class

**Console display** (Rich library):

| Method | Content |
|--------|---------|
| `display_full_report()` | Top picks → Sector overview → News alerts → Signals → Shorts |
| `display_suggestions()` | Ranked stocks with scores and reasons |
| `display_shortlist()` | Buy / Short / Trending candidates |
| `display_stock_analysis()` | Single-stock deep dive with LLM insights |
| `display_portfolio()` | Open positions with unrealized PnL |
| `display_gain_report()` | Closed trade statistics |

**Signal colors:**

| Signal | Color |
|--------|-------|
| STRONG BUY | Bold green |
| BUY | Green |
| HOLD | Yellow |
| SELL | Red |
| STRONG SELL | Bold red |

**Export methods:**
- `export_csv(signals)` → `data/processed/predictions_YYYY-MM-DD.csv`
- `export_json(signals)` → `data/processed/predictions_YYYY-MM-DD.json`
- `export_gain_report(report)` → `data/trades/report_YYYY-MM-DD.json`

**CSV columns:** Symbol, Name, Signal, Confidence, BUY%, HOLD%, SELL%, Outlook, Probabilities, Technical summary

---

## 11. Configuration System

### settings.yaml Structure

**File:** `config/settings.yaml` (89 lines)

```yaml
data:
  provider: yfinance
  default_start_date: "2020-01-01"
  cache_dir: data/raw

news:
  rss_base_url: "https://news.google.com/rss/search"
  queries: ["Indian stock market", "NSE BSE stocks", "NIFTY 50", "Indian economy"]
  max_articles_per_query: 50
  cache_dir: data/news_cache
  cache_expiry_hours: 6

sentiment:
  model_name: "ProsusAI/finbert"
  batch_size: 32

ner:
  spacy_model: "en_core_web_sm"

llm:
  provider: ollama
  ollama:
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"
    timeout: 120
  cache_dir: data/news_cache
  cache_expiry_hours: 24

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

signals:
  confidence_threshold: 0.6
  strong_threshold: 0.8
  short_confidence_threshold: 0.7
  buy_return_threshold: 0.01
  sell_return_threshold: -0.01

screener:
  volume_spike_threshold: 1.5
  news_volume_spike_threshold: 2.0
  news_lookback_days: 3

report:
  export_dir: data/processed
  formats: ["console", "csv", "json"]

paper_trading:
  ledger_file: data/trades/ledger.json
  report_dir: data/trades
```

### nifty50.yaml

**File:** `config/nifty50.yaml` (76 lines)

Defines 50 companies across 11 sectors:
- IT, Banking, Financial_Services, Oil_Gas_Energy, Pharma_Healthcare
- Automobile, FMCG, Metals_Mining, Cement_Construction, Telecom, Conglomerate

Each entry: `{symbol: "RELIANCE.NS", name: "Reliance Industries"}`

### Config Loader

**File:** `src/stock_prediction/config.py`

- Loads YAML with caching (single read per session)
- Accessed throughout the codebase via `get_settings()` helper

---

## 12. Data Flow Diagrams

### Training Pipeline

```
stockpredict train --symbols RELIANCE.NS
  │
  ├─ FeaturePipeline.build_features()
  │    ├─ YFinanceProvider.fetch_historical()  →  OHLCV DataFrame
  │    ├─ add_technical_indicators()           →  +22 technical columns
  │    ├─ NewsFeatureGenerator.generate()      →  +18 news columns (optional)
  │    ├─ BrokerNewsAnalyzer.analyze_stock()   →  +10 LLM columns (optional)
  │    └─ Label creation (return_1d → signal)  →  +3 label columns
  │
  ├─ prepare_training_data()
  │    ├─ Rolling 60-day windows → LSTM sequences (N, 60, F)
  │    ├─ Last row of windows → XGBoost tabular (N, F)
  │    └─ raises ValueError if:
  │         ├─ yfinance returned no price data (wrong ticker / network)
  │         └─ < 70 rows after dropna (need ~130+ raw trading days)
  │
  ├─ [on ValueError]  → status=no_data, reason=<specific message>
  ├─ [on Exception]   → status=failed,  reason=<exception message>
  │
  ├─ Chronological split (80% train / 20% val)
  │
  ├─ StandardScaler.fit_transform(train), transform(val)
  │
  ├─ [Classification models — LSTM, XGBoost]
  │    ├─ LSTMPredictor.train(X_seq_train, y_train)     → CrossEntropyLoss + early stopping
  │    └─ XGBoostPredictor.train(X_tab_train, y_train)  → multi:softprob + early stopping
  │
  ├─ [Regression models — Encoder-Decoder, TFT]
  │    ├─ prepare_regression_data() → (sequences, reg_targets, labels)
  │    ├─ EncoderDecoderPredictor.train(X_seq, reg_targets)  → MSE loss; tuned by MAPE
  │    └─ TFTPredictor.train(X_seq, reg_targets)             → MSE loss; tuned by MAPE
  │
  ├─ [RL agents — Q-learning, DQN]
  │    ├─ QLearningPredictor.train(X_last_step, returns_1d)  → ε-greedy Q-table update
  │    └─ DQNPredictor.train(X_last_step, returns_1d)        → Huber loss + replay buffer
  │
  ├─ [Prophet]
  │    └─ ProphetPredictor.tune() + fit_full()               → changepoint grid search
  │
  ├─ Ensemble weights: each model's val balanced_accuracy / total
  │
  ├─ Retrain all selected models on full dataset (100%)
  │
  ├─ Save to data/models/RELIANCE_NS/
  │    ├─ lstm.pt / xgboost.joblib / encoder_decoder.pt
  │    ├─ prophet.joblib / tft.pt / qlearning.joblib / dqn.pt
  │    └─ meta.joblib (scalers, feature_names, weights, trained_at, horizon)
  │
  └─ Write data/reports/train_summary.csv
       └─ Symbol | Status | Val Accuracy | Reason  (one row per symbol)
```

### Prediction Pipeline

```
stockpredict predict --symbols RELIANCE.NS
  │
  ├─ ModelTrainer.load_models("RELIANCE.NS")
  │    ├─ Load lstm.pt, xgboost.joblib, meta.joblib
  │    └─ Check staleness (warn if > 30 days)
  │
  ├─ FeaturePipeline.build_features() [latest data]
  │
  ├─ Scale features with saved StandardScaler
  │
  ├─ EnsembleModel.predict(X_seq, X_tab)
  │    ├─ LSTM / Encoder-Decoder / TFT: predict_proba(X_seq) → (N, 3)
  │    ├─ XGBoost: predict_proba(X_tab) → (N, 3)
  │    ├─ Q-learning / DQN: predict_proba(X_seq[:, -1, :]) → (N, 3)
  │    ├─ Prophet: broadcast single horizon-ahead probability → (N, 3)
  │    └─ Weighted average using per-model val balanced accuracy weights
  │
  ├─ SignalGenerator.generate()
  │    ├─ Apply confidence thresholds (0.6 / 0.8)
  │    └─ Compute short score (sell_prob + RSI + MACD + SMA50)
  │
  ├─ StockScreener.screen() [optional]
  │
  └─ ReportFormatter.display_full_report()
       └─ export_csv() / export_json() [if --export]
```

### Screening Pipeline

```
stockpredict suggest --count 10
  │
  ├─ StockScreener.suggest(count=10)
  │    ├─ For each NIFTY 50 stock:
  │    │    ├─ Fetch 60-day OHLCV
  │    │    ├─ Compute technical indicators
  │    │    ├─ Calculate momentum (1w*0.6 + 1m*0.4)
  │    │    ├─ Check volume spike, RSI, SMA crossover, 52w high
  │    │    └─ Count news mentions (via NER entity linking)
  │    ├─ Score and rank all stocks
  │    └─ Return top N with reasons
  │
  └─ ReportFormatter.display_suggestions()
```

---

## 13. Storage and Persistence Layout

```
stock_prediction/
├── config/
│   ├── settings.yaml              # All configuration parameters
│   └── nifty50.yaml               # NIFTY 50 sector definitions
├── data/
│   ├── raw/                       # OHLCV cache from yfinance
│   ├── models/                    # Trained ML models
│   │   └── {SYMBOL}/             # e.g., RELIANCE_NS/
│   │       ├── lstm.pt            # PyTorch LSTM state_dict + metadata
│   │       ├── xgboost.joblib     # XGBClassifier + feature_names
│   │       ├── encoder_decoder.pt # PyTorch Encoder-Decoder state_dict
│   │       ├── prophet.joblib     # Fitted Prophet model + regressors
│   │       ├── tft.pt             # PyTorch TFT state_dict + metadata
│   │       ├── qlearning.joblib   # Q-table + bin edges + state feature indices
│   │       ├── dqn.pt             # PyTorch DQN Q-network state_dict + metadata
│   │       └── meta.joblib        # Scalers, feature_names, weights, trained_at
│   ├── news_cache/                # RSS + LLM response cache
│   │   ├── {md5_query}.json       # RSS articles (6-hour expiry)
│   │   └── {md5_symbol_date}.json # LLM scores (24-hour expiry)
│   ├── reports/                   # Per-run stage CSV outputs (gitignored, overwritten each run)
│   │   ├── train_summary.csv      # Per-symbol training status, accuracy, failure reason
│   │   ├── signals.csv            # Trading signals (predict/screen)
│   │   ├── short_candidates.csv   # Short selling candidates
│   │   ├── top_picks.csv          # Pre-screened top picks
│   │   ├── sector_momentum.csv    # Sector leaders
│   │   ├── news_alerts.csv        # News-discovered stocks
│   │   ├── shortlist.csv          # Buy / Short / Trending candidates
│   │   ├── suggestions.csv        # Ranked momentum watchlist
│   │   ├── analyze.csv            # Broker analysis scores (single stock)
│   │   ├── portfolio.csv          # Open paper trading positions
│   │   ├── gain_summary.csv       # Paper trading gain/loss summary
│   │   └── gain_per_stock.csv     # Per-symbol gain/loss breakdown
│   ├── processed/                 # Export outputs
│   │   ├── predictions_YYYY-MM-DD.csv
│   │   └── predictions_YYYY-MM-DD.json
│   └── trades/                    # Paper trading data
│       ├── ledger.json            # Trade history (all trades)
│       └── report_YYYY-MM-DD.json # Gain report snapshots
```

### File Format Summary

| Artifact | Format | Serialization |
|----------|--------|---------------|
| LSTM / Encoder-Decoder / TFT / DQN | `.pt` | `torch.save(state_dict + metadata)` |
| XGBoost / Prophet / Q-learning | `.joblib` | `joblib.dump(model + metadata)` |
| Model metadata | `.joblib` | `joblib.dump(scalers + feature_names + weights + trained_at)` |
| News cache | `.json` | JSON array of NewsArticle dicts |
| LLM cache | `.json` | JSON dict of scores + `_summary` |
| Trade ledger | `.json` | JSON array of PaperTrade dicts |
| Predictions export | `.csv`/`.json` | pandas/json serialization |
| Gain reports | `.json` | JSON serialization of GainReport |

---

## 14. External Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2+ | DataFrames, time series |
| numpy | 1.26+ | Numerical operations |
| yfinance | 0.2.36+ | Yahoo Finance OHLCV data |
| scikit-learn | 1.4+ | StandardScaler, metrics |
| xgboost | 2.0+ | Gradient boosted trees |
| torch | 2.2+ | LSTM neural network |
| joblib | 1.3+ | Model serialization |
| transformers | 4.38+ | FinBERT sentiment model |
| spacy | 3.7+ | NER (en_core_web_sm) |
| ollama | 0.4+ | Local LLM client |
| ta | 0.11+ | Technical analysis indicators |
| feedparser | 6.0+ | RSS feed parsing |
| beautifulsoup4 | 4.12+ | HTML snippet extraction |
| requests | 2.31+ | HTTP requests |
| click | 8.1+ | CLI framework |
| rich | 13.7+ | Console formatting & tables |
| pyyaml | 6.0+ | YAML configuration |
| tqdm | 4.66+ | Progress bars |

### External Services

| Service | Usage | Required |
|---------|-------|----------|
| Yahoo Finance | OHLCV data via yfinance | Yes |
| Google News RSS | Market news articles | Optional (--no-news) |
| Ollama (local) | LLM broker analysis | Optional (--no-llm) |

### NLP Models (Downloaded on First Use)

| Model | Source | Size |
|-------|--------|------|
| ProsusAI/finbert | Hugging Face | ~250MB |
| en_core_web_sm | spaCy | ~12MB |
| llama3.1:8b | Ollama | ~4.7GB |

---

## 15. Extensibility Points

### Adding a New Data Provider

1. Create `src/stock_prediction/data/new_provider.py`
2. Implement `DataProvider` ABC (`fetch_historical`, `fetch_batch`, `fetch_latest`)
3. Register in `data/__init__.py` factory (`get_provider`)

### Adding a New LLM Provider

1. Create `src/stock_prediction/llm/new_provider.py`
2. Implement `LLMProvider` ABC (`analyze`, `analyze_batch`, `is_available`)
3. Register in `llm/__init__.py` factory (`get_llm_provider`)

### Adding New Technical Indicators

1. Edit `src/stock_prediction/features/technical.py`
2. Add indicator computation using the `ta` library
3. Update `settings.yaml` with configurable parameters

### Adding a New ML Model

1. Create model class in `src/stock_prediction/models/`
2. Implement `train()`, `predict_proba()`, `predict()`, `save()`, `load()`
3. Update `EnsembleModel` to include new model weights
4. Update `ModelTrainer` training and loading logic

### Adding New Screener Criteria

1. Edit `src/stock_prediction/signals/screener.py`
2. Add scoring logic in `_score_stock()` method
3. Update thresholds in `settings.yaml` under `screener`

### Adding New News Sources

1. Create a new fetcher class in `src/stock_prediction/news/`
2. Implement same interface as `GoogleNewsRSSFetcher` (return `NewsArticle` list)
3. Integrate in `NewsFeatureGenerator`

### Adding New Export Formats

1. Add method in `src/stock_prediction/signals/report.py`
2. Register format in `settings.yaml` under `report.formats`
