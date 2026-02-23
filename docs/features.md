# Features Guide

## Indian Stock Market Prediction System

---

## 1. ML Prediction Signals

The system generates trading signals using an ensemble of two ML models trained on historical price data, technical indicators, news sentiment, and LLM-derived insights.

### Signal Types

| Signal | Meaning | Condition |
|--------|---------|-----------|
| **STRONG BUY** | High-confidence buy | BUY with confidence >= 80% |
| **BUY** | Model predicts upward movement | BUY with confidence >= 60% |
| **HOLD** | No clear direction or low confidence | HOLD signal, or confidence < 60% |
| **SELL** | Model predicts downward movement | SELL with confidence >= 60% |
| **STRONG SELL** | High-confidence sell | SELL with confidence >= 80% |

Each signal includes:
- **Confidence score** (0-100%) — how certain the model is
- **Probability breakdown** — BUY%, HOLD%, SELL% probabilities
- **Short score** (0-1) — suitability for short selling
- **Weekly/monthly outlook** — directional summary
- **Technical summary** — key indicator readings

### Ensemble Model

- **LSTM neural network** (40% weight) — captures sequential price patterns over 60-day windows
- **XGBoost** (60% weight) — captures tabular feature relationships
- Combined via weighted probability averaging

---

## 2. Technical Analysis

22+ technical indicators are automatically computed from OHLCV data:

### Momentum
- **RSI** (14-period) — Relative Strength Index, identifies overbought (>70) / oversold (<30)

### Trend
- **MACD** (12, 26, 9) — Moving Average Convergence Divergence with signal line and histogram
- **SMA** (20-day, 50-day) — Simple Moving Averages
- **EMA** (12-day, 26-day) — Exponential Moving Averages
- **SMA 20/50 Crossover** — Bullish/bearish crossover detection

### Volatility
- **Bollinger Bands** (20-period, 2 std) — Upper, middle, lower bands + bandwidth
- **ATR** (14-period) — Average True Range

### Volume
- **OBV** — On-Balance Volume (accumulation/distribution)
- **VWAP** — Volume-Weighted Average Price
- **Volume Ratio** — Current volume vs 20-day average

### Oscillators
- **Stochastic %K and %D** (14-period)

### Derived
- Price-to-SMA ratios (20-day, 50-day)
- Volume SMA (20-day)

---

## 3. News Sentiment Analysis

Real-time news analysis using Google News RSS feeds and FinBERT.

### How It Works
1. Fetches articles from Google News RSS for Indian stock market queries
2. Links articles to specific stocks using NER (spaCy) + company alias matching
3. Analyzes sentiment using **FinBERT** (financial domain BERT model)
4. Aggregates sentiment across 1-day, 7-day, and 30-day windows

### Sentiment Features
- **Mean sentiment** — average positive/negative lean
- **Sentiment volatility** — how much opinions vary
- **News volume** — article count (higher = more attention)
- **Positive/negative ratio** — proportion of bullish vs bearish articles
- **Sentiment trend** — directional slope over 7-day and 30-day periods
- **Category detection** — earnings, mergers, regulation, management changes, dividends, expansion

### Disable
Use `--no-news` flag on any command to run with technical indicators only.

---

## 4. LLM Broker Insights

Local LLM analysis (via Ollama) that simulates broker research desk commentary.

### 10-Dimension Analysis (0-10 Scale)
| Dimension | What It Measures |
|-----------|-----------------|
| Earnings Outlook | Expected earnings trajectory |
| Competitive Position | Market dominance and moat |
| Management Quality | Leadership track record |
| Sector Momentum | Industry-level trends |
| Risk Level | Downside risk factors |
| Growth Catalyst | Upcoming growth triggers |
| Valuation Signal | Under/overvaluation signal |
| Institutional Interest | Institutional buying patterns |
| Macro Impact | Macroeconomic sensitivity |
| Overall Broker Score | Composite recommendation |

Each analysis includes a 1-2 sentence summary. Results are cached for 24 hours.

### Requirements
- Ollama running locally (`http://localhost:11434`)
- Model: `llama3.1:8b` (default, configurable)

### Disable
Use `--no-llm` flag to skip LLM analysis.

---

## 5. Stock Discovery

### suggest — Top Momentum Stocks

Ranks all NIFTY 50 stocks by a composite score considering:
- Recent momentum (1-week and 1-month returns)
- Volume spikes (>1.5x average)
- RSI extremes (oversold < 30, overbought > 70)
- SMA 20/50 bullish crossover
- Proximity to 52-week high (within 5%)
- News mention frequency

```
stockpredict suggest --count 10
stockpredict suggest --count 20 --no-news
```

### shortlist — Buy/Short/Trending Candidates

Three categories of actionable stocks:

- **Buy candidates** — Top N NIFTY 50 stocks by composite score (best opportunities)
- **Short candidates** — Bottom N stocks showing weakness (negative momentum, overbought RSI, bearish technicals)
- **Trending** — Non-NIFTY stocks discovered from recent news (potential breakout names)

```
stockpredict shortlist --count 5
stockpredict shortlist --count 10 --no-news --no-llm
```

### screen — Full Stock Screener

Comprehensive screening without requiring trained models:
- **Top picks** — stocks with volume spikes or strong technical signals
- **Sector leaders** — best-performing stock in each sector
- **News alerts** — stocks with breaking news activity
- **Full rankings** — all screened stocks sorted by score

```
stockpredict screen
stockpredict screen --symbols RELIANCE.NS,TCS.NS,INFY.NS
```

---

## 6. Paper Trading

Simulated trading with real-time prices. No actual money is involved.

### Open a Long Position (Buy)

```
stockpredict test-buy --symbol RELIANCE.NS --amount 50000
```

Buys shares worth the specified amount (INR) at the current market price.

### Close a Long Position (Sell)

```
stockpredict test-sell --symbol RELIANCE.NS
stockpredict test-sell --symbol RELIANCE.NS --trade-id abc12345
```

Closes the position and calculates realized P&L.

### Open a Short Position

```
stockpredict test-short --symbol RELIANCE.NS --amount 50000
```

Opens a short sell position (profit when price drops).

### View Portfolio

```
stockpredict test-portfolio
```

Shows all open positions with current prices and unrealized P&L.

### Gain/Loss Report

```
stockpredict test-calculate-gain
stockpredict test-calculate-gain --export
```

Shows:
- Total trades, win/loss count
- Total P&L (INR and %)
- Best and worst trades
- Per-stock breakdown
- Open positions with unrealized P&L

Export saves to `data/trades/report_YYYY-MM-DD.json`.

---

## 7. Export Formats

### CSV Export

```
stockpredict predict --symbols RELIANCE.NS,TCS.NS --export
```

Saves to `data/processed/predictions_YYYY-MM-DD.csv` with columns:
- Symbol, Name, Signal, Confidence
- BUY%, HOLD%, SELL%
- Outlook, Technical Summary

### JSON Export

Same `--export` flag also generates `data/processed/predictions_YYYY-MM-DD.json` with full structured data including all probabilities and metadata.

### Gain Report Export

```
stockpredict test-calculate-gain --export
```

Saves to `data/trades/report_YYYY-MM-DD.json`.

---

## 8. CLI Commands Reference

**Global option:** `--log-level {DEBUG, INFO, WARNING, ERROR}` (default: INFO)

### Data & Training

| Command | Description | Key Options |
|---------|-------------|-------------|
| `fetch-data` | Download and cache OHLCV data | `--symbols, -s` (comma-separated), `--start-date` |
| `train` | Train LSTM + XGBoost models | `--symbols, -s`, `--start-date`, `--end-date`, `--no-news`, `--no-llm` |

### Prediction & Analysis

| Command | Description | Key Options |
|---------|-------------|-------------|
| `predict` | Generate trading signals with screener | `--symbols, -s`, `--export`, `--no-news`, `--no-llm` |
| `analyze` | Deep single-stock analysis with LLM | `--symbol, -s` (required), `--no-news`, `--no-llm` |

### Stock Discovery

| Command | Description | Key Options |
|---------|-------------|-------------|
| `suggest` | Top momentum stocks | `--count, -n` (default: 10), `--no-news` |
| `shortlist` | Buy/short/trending candidates | `--count, -n` (default: 5), `--no-news`, `--no-llm` |
| `screen` | Full stock screener | `--symbols, -s`, `--no-news`, `--no-llm` |

### Paper Trading

| Command | Description | Key Options |
|---------|-------------|-------------|
| `test-buy` | Buy (open long) | `--symbol, -s`, `--amount, -a` (INR) |
| `test-sell` | Sell (close position) | `--symbol, -s`, `--trade-id` (optional) |
| `test-short` | Short sell (open short) | `--symbol, -s`, `--amount, -a` (INR) |
| `test-portfolio` | View open positions | — |
| `test-calculate-gain` | P&L report | `--export` |

### Typical Workflow

```bash
# 1. Discover interesting stocks
stockpredict suggest --count 10

# 2. Train models for selected stocks
stockpredict train --symbols RELIANCE.NS,TCS.NS,INFY.NS

# 3. Get predictions
stockpredict predict --symbols RELIANCE.NS,TCS.NS,INFY.NS --export

# 4. Deep dive on a specific stock
stockpredict analyze --symbol RELIANCE.NS

# 5. Paper trade based on signals
stockpredict test-buy --symbol RELIANCE.NS --amount 100000

# 6. Monitor portfolio
stockpredict test-portfolio

# 7. Check gains
stockpredict test-calculate-gain
```

---

## 9. Configuration Options

All settings are in `config/settings.yaml`. Key user-tunable options:

### Data
| Setting | Default | Description |
|---------|---------|-------------|
| `data.provider` | `yfinance` | Data source |
| `data.default_start_date` | `2020-01-01` | Historical data start |

### News & NLP
| Setting | Default | Description |
|---------|---------|-------------|
| `news.max_articles_per_query` | `50` | Max articles fetched per query |
| `news.cache_expiry_hours` | `6` | News cache freshness |
| `llm.ollama.model` | `llama3.1:8b` | Ollama model name |
| `llm.ollama.base_url` | `http://localhost:11434` | Ollama server URL |
| `llm.cache_expiry_hours` | `24` | LLM cache freshness |

### Model Training
| Setting | Default | Description |
|---------|---------|-------------|
| `models.lstm.epochs` | `50` | Max training epochs |
| `models.lstm.patience` | `10` | Early stopping patience |
| `models.xgboost.n_estimators` | `500` | Number of trees |
| `models.ensemble.lstm_weight` | `0.4` | LSTM contribution |
| `models.ensemble.xgboost_weight` | `0.6` | XGBoost contribution |
| `models.staleness_warning_days` | `30` | Warn if model is older |
| `features.sequence_length` | `60` | LSTM lookback window (days) |

### Signal Thresholds
| Setting | Default | Description |
|---------|---------|-------------|
| `signals.confidence_threshold` | `0.6` | Minimum confidence for BUY/SELL |
| `signals.strong_threshold` | `0.8` | Threshold for STRONG signals |
| `signals.buy_return_threshold` | `0.01` | 1% return labels as BUY |
| `signals.sell_return_threshold` | `-0.01` | -1% return labels as SELL |

### Screener
| Setting | Default | Description |
|---------|---------|-------------|
| `screener.volume_spike_threshold` | `1.5` | Volume spike multiplier |
| `screener.news_lookback_days` | `3` | Days of news to consider |

### Paper Trading
| Setting | Default | Description |
|---------|---------|-------------|
| `paper_trading.ledger_file` | `data/trades/ledger.json` | Trade history file |
| `paper_trading.report_dir` | `data/trades` | Report output directory |

---

## 10. Supported Stocks

The system covers all **NIFTY 50** stocks across 11 sectors:

| Sector | Example Stocks |
|--------|---------------|
| IT | TCS, Infosys, Wipro, HCL Tech, Tech Mahindra |
| Banking | HDFC Bank, ICICI Bank, SBI, Kotak, Axis, IndusInd |
| Financial Services | Bajaj Finance, Bajaj Finserv, HDFC Life, SBI Life |
| Oil, Gas & Energy | Reliance, ONGC, NTPC, Power Grid, Adani Energy |
| Pharma & Healthcare | Sun Pharma, Dr. Reddy's, Cipla, Apollo, Divi's |
| Automobile | M&M, Tata Motors, Maruti, Bajaj Auto, Eicher, Hero |
| FMCG | Hindustan Unilever, ITC, Nestle, Tata Consumer, Britannia |
| Metals & Mining | Tata Steel, JSW Steel, Hindalco, Adani Enterprises |
| Cement & Construction | UltraTech, Grasim, L&T |
| Telecom | Bharti Airtel |
| Conglomerate | ITC (diversified) |

The **trending** feature in `shortlist` can also discover non-NIFTY stocks from news.
