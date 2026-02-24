# Stock Prediction System — Usage Guide

## Prerequisites

Activate the conda environment before running any commands.

```bash
conda activate stock_prediction
```

## Typical Workflow

```
shortlist → fetch-data → train → predict → analyze → paper trade
```

> **Note:** `fetch-data` is a data preview step only — it does not save data to disk.
> `train` fetches data live from yfinance independently. See each step below for details.

---

## Step 1: Discover What to Trade

### `shortlist` — Buy / Short / Trending Candidates

**What it does:**
Screens all NIFTY 50 stocks without requiring any trained ML models. For each stock it fetches the last 60 days of price data, computes 22+ technical indicators, and calculates a composite score based on momentum, volume, RSI, SMA crossovers, 52-week high proximity, and news mentions. It then produces three lists:

- **Buy candidates** — Top N NIFTY 50 stocks by composite score (best momentum + technicals)
- **Short candidates** — Bottom N stocks with weak technicals (overbought RSI, below SMA50, bearish MACD, negative momentum)
- **Trending** — Non-NIFTY stocks discovered from recent news via NER entity linking (potential breakout names not in NIFTY 50)

When `--no-news` is omitted (default), it also fetches Google News RSS articles, runs FinBERT sentiment analysis, and uses spaCy NER to link articles to stock tickers. The `--no-llm` flag controls whether an Ollama LLM is used for deeper news discovery in the trending section.

**Dependencies:** None. Runs independently without trained models.

**Output:**
- Console: Three Rich tables — Buy Candidates, Short Candidates, Trending from News
- File: `data/reports/shortlist.csv` — all rows combined, with a `Category` column (Buy / Short / Trending). Overwritten on each run.

```bash
stockpredict shortlist -n 5                  # Top 5 buy + short candidates + trending from news
stockpredict shortlist -n 3 --no-news        # Technical-only (faster, no trending section)
stockpredict shortlist -n 5 --no-llm         # News alias matching only, skip LLM discovery
```

---

### `suggest` — Ranked Watchlist

**What it does:**
Similar to `shortlist` but produces a single ranked list of NIFTY 50 stocks rather than separate buy/short/trending buckets. Each stock receives a composite score (momentum + volume + RSI + crossover + 52-week high + news mentions) and the top N are returned with their scores and the reasons they ranked.

**Dependencies:** None. Runs independently without trained models.

**Output:**
- Console: One Rich table — Suggested Stocks ranked by score
- File: `data/reports/suggestions.csv` — all ranked stocks. Overwritten on each run.

```bash
stockpredict suggest -n 10                   # Top 10 stocks ranked by momentum + news
stockpredict suggest -n 5 --no-news          # Technical-only ranking
```

---

## Step 2: Fetch Data

### `fetch-data` — Data Availability Check

**What it does:**
Fetches historical OHLCV price data from yfinance for the specified symbols (or all NIFTY 50 by default) and prints a summary of how many rows were retrieved and the date range covered per stock. This is useful for verifying connectivity to yfinance and confirming that data exists for your chosen symbols before committing to a long training run.

> **Important:** This command does **not** save data to disk. The data is fetched into memory, summarised to the console, and then discarded. The `train` command fetches data independently from yfinance at training time. `fetch-data` is a diagnostic / preview tool only.

**Dependencies:** None.

**Output:**
- Console only: Per-symbol row count and date range (green = data found, red = no data)
- Files: None

```bash
stockpredict fetch-data                                  # All NIFTY 50
stockpredict fetch-data -s RELIANCE.NS,TCS.NS,SBIN.NS   # Specific stocks
stockpredict fetch-data -s RELIANCE.NS --start-date 2023-01-01
```

---

## Step 3: Train Models

### `train` — Train LSTM + XGBoost Ensemble

**What it does:**
Trains a per-symbol ML ensemble (LSTM neural network + XGBoost classifier) for each specified stock. For each symbol the pipeline:

1. Fetches OHLCV data live from yfinance (from `start-date` or `2020-01-01` by default)
2. Computes 22+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Stochastic, SMA/EMA crossovers)
3. Optionally fetches Google News RSS articles, runs FinBERT sentiment analysis and spaCy NER to produce 18 news features
4. Optionally calls the local Ollama LLM to generate 10 broker-style insight scores per stock
5. Creates training labels from next-day returns (≥1% → BUY, ≤-1% → SELL, else HOLD)
6. Builds 60-day rolling windows for the LSTM and tabular snapshots for XGBoost
7. Splits chronologically (80% train / 20% validation — no shuffling to prevent data leakage)
8. Trains the LSTM (Adam + CrossEntropyLoss, up to 50 epochs, early stopping patience=10)
9. Trains XGBoost (multi:softprob, up to 500 trees, early stopping after 20 rounds)
10. Saves all model artefacts to `data/models/{SYMBOL}/`

> **Note:** `train` fetches its own data directly from yfinance. Running `fetch-data` beforehand is not required and has no effect on training.

**Dependencies:** None on prior steps. Requires internet access to yfinance and optionally Google News RSS and a running Ollama server.

**Output:**
- Console: Training progress and per-stock success/failure summary
- Files written per symbol to `data/models/{SYMBOL}/`:
  - `lstm.pt` — PyTorch LSTM weights and architecture metadata
  - `xgboost.joblib` — trained XGBoost classifier with feature names
  - `meta.joblib` — fitted StandardScalers, feature names, and `trained_at` timestamp

```bash
stockpredict train -s RELIANCE.NS,TCS.NS,SBIN.NS        # Specific stocks
stockpredict train                                        # All NIFTY 50
stockpredict train -s RELIANCE.NS --no-news --no-llm     # Technical-only (faster)
stockpredict train -s RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-31
```

---

## Step 4: Generate Predictions

### `predict` — Trading Signals from Trained Models

**What it does:**
Loads the trained models for each symbol and generates BUY / HOLD / SELL signals with confidence scores. For each symbol it:

1. Loads `lstm.pt`, `xgboost.joblib`, and `meta.joblib` from `data/models/{SYMBOL}/` — warns if the model is more than 30 days old
2. Fetches the latest price data from yfinance and builds current features (same pipeline as training)
3. Scales features using the saved StandardScaler
4. Runs the LSTM and XGBoost, then combines via weighted ensemble (40% LSTM + 60% XGBoost)
5. Applies confidence thresholds: signals below 60% confidence are downgraded to HOLD; signals above 80% are upgraded to STRONG BUY / STRONG SELL
6. Computes a short score (sell probability + RSI + MACD + SMA50) and flags short-selling candidates
7. Runs the stock screener to produce top picks, sector momentum, and news alerts alongside the model signals

**Dependencies:** Trained model files must exist in `data/models/{SYMBOL}/` — run `train` first for each symbol you want to predict.

**Output:**
- Console: Five Rich tables — Top Picks, Sector Momentum, News Alerts, Trading Signals, Short Selling Candidates
- Files (always written to `data/reports/`, overwritten each run):
  - `signals.csv` — full signal table (Symbol, Name, Signal, Confidence, BUY%, HOLD%, SELL%, Outlook)
  - `short_candidates.csv` — short-selling candidates with scores and reasons
  - `top_picks.csv` — pre-screened top picks
  - `sector_momentum.csv` — sector leaders with 1W and 1M returns
  - `news_alerts.csv` — non-NIFTY stocks with breaking news
- Files (only with `--export` flag, date-stamped, not overwritten):
  - `data/processed/predictions_YYYY-MM-DD.csv` — full signal export
  - `data/processed/predictions_YYYY-MM-DD.json` — full signal export in JSON

```bash
stockpredict predict -s RELIANCE.NS,TCS.NS               # Specific stocks
stockpredict predict                                      # All NIFTY 50
stockpredict predict -s RELIANCE.NS --export              # Export results to CSV + JSON
stockpredict predict --no-news --no-llm                   # Technical-only predictions
```

---

## Step 5: Deep Dive (Single Stock Analysis)

### `analyze` — Detailed Single-Stock Analysis

**What it does:**
Produces a comprehensive report for a single stock combining the ML signal, technical indicator readings, LLM broker insights, and recent news headlines. It loads the trained model, generates a prediction, then sends the top 15 recent headlines to the local Ollama LLM which scores the stock across 10 dimensions (earnings outlook, competitive position, management quality, sector momentum, risk level, growth catalyst, valuation signal, institutional interest, macro impact, overall broker score) and produces a short narrative summary. LLM results are cached for 24 hours.

**Dependencies:** A trained model must exist for the symbol in `data/models/{SYMBOL}/` — run `train` first.

**Output:**
- Console: Signal summary, weekly/monthly outlook, LLM narrative summary, Broker Analysis Scores table, recent headlines
- File: `data/reports/analyze.csv` — broker score dimensions and values (written only when LLM scores are available). Overwritten on each run.

```bash
stockpredict analyze -s RELIANCE.NS                      # Full analysis
stockpredict analyze -s RELIANCE.NS --no-llm             # Skip LLM broker analysis
stockpredict analyze -s RELIANCE.NS --no-news            # Skip news features
```

---

## Step 6: Run the Screener

### `screen` — Full Stock Screener (No Model Required)

**What it does:**
Runs the complete stock screener across the specified symbols (or all NIFTY 50) without requiring any trained ML models. It fetches price data, computes technical indicators, ranks stocks within their sectors, identifies volume spikes and technical breakouts, discovers news-mentioned stocks, and produces a structured four-section report. This is useful for getting a broad market overview independent of the ML pipeline.

**Dependencies:** None. Does not require trained models.

**Output:**
- Console: Four Rich tables — Top Picks (Pre-Screened), Sector Momentum, News Alerts, Trading Signals (screener-based)
- Files (always, `data/reports/`, overwritten each run):
  - `top_picks.csv`
  - `sector_momentum.csv`
  - `news_alerts.csv`
  - `signals.csv`
  - `short_candidates.csv`

```bash
stockpredict screen                                      # Full NIFTY 50 screen
stockpredict screen -s RELIANCE.NS,TCS.NS,INFY.NS       # Specific stocks
stockpredict screen --no-news                            # Disable news discovery
stockpredict screen --no-llm                             # Disable LLM discovery
```

---

## Step 7: Paper Trading

Simulate trades without real money to test your strategy. All positions and trade history are persisted in `data/trades/ledger.json` and updated on every operation.

### `test-buy` — Open a Long Position

**What it does:**
Fetches the current market price for the symbol from yfinance, calculates the number of shares purchasable for the given amount (INR), and records an OPEN LONG trade in the ledger. If an existing open SHORT position exists for the symbol it covers that instead.

**Dependencies:** None on prior steps, but practically used after reviewing signals from `predict` or `analyze`.

**Output:**
- Console: Confirmation with entry price, quantity, and trade ID
- File: `data/trades/ledger.json` — updated with the new trade

```bash
stockpredict test-buy -s RELIANCE.NS -a 50000
```

---

### `test-short` — Open a Short Position

**What it does:**
Fetches the current price, records an OPEN SHORT trade in the ledger at that price. Profit is realised when the price drops below the entry price at close time.

**Dependencies:** None on prior steps.

**Output:**
- Console: Confirmation with entry price, quantity, and trade ID
- File: `data/trades/ledger.json` — updated

```bash
stockpredict test-short -s TCS.NS -a 30000
```

---

### `test-sell` — Close a Position

**What it does:**
Fetches the current market price, closes the most recent (or specified) open position for the symbol, calculates realised PnL `(exit - entry) × qty` for LONG or `(entry - exit) × qty` for SHORT, and marks the trade CLOSED in the ledger.

**Dependencies:** An open position must exist for the symbol (created via `test-buy` or `test-short`).

**Output:**
- Console: Confirmation with exit price and realised PnL (INR and %)
- File: `data/trades/ledger.json` — trade updated to CLOSED status

```bash
stockpredict test-sell -s RELIANCE.NS
stockpredict test-sell -s TCS.NS --trade-id abc123       # Close specific trade
```

---

### `test-portfolio` — View Open Positions

**What it does:**
Reads all OPEN trades from the ledger, fetches current market prices for each symbol, and computes unrealized PnL for every position. Displays a summary of total capital deployed and aggregate unrealized gain/loss.

**Dependencies:** Open positions must exist in the ledger.

**Output:**
- Console: Rich table of all open positions with entry price, current price, quantity, amount, PnL, and PnL %
- File: `data/reports/portfolio.csv` — same data in CSV. Overwritten on each run.

```bash
stockpredict test-portfolio
```

---

### `test-calculate-gain` — Gain / Loss Report

**What it does:**
Reads all CLOSED trades from the ledger and computes aggregate statistics: total trades, win/loss count, win rate, total PnL (INR and %), best and worst individual trades, and a per-symbol breakdown. Also reads OPEN trades and computes current unrealized PnL by fetching live prices.

**Dependencies:** Closed positions must exist in the ledger (from `test-sell`).

**Output:**
- Console: Summary metrics table, best/worst trade callouts, Per-Stock Breakdown table
- Files (always, `data/reports/`, overwritten each run):
  - `gain_summary.csv` — overall metrics (Metric, Value)
  - `gain_per_stock.csv` — per-symbol breakdown (Symbol, Trades, PnL, PnL %)
- File (only with `--export` flag, date-stamped):
  - `data/trades/report_YYYY-MM-DD.json` — full gain report in JSON

```bash
stockpredict test-calculate-gain
stockpredict test-calculate-gain --export                # Export to JSON
```

---

## Dependency Summary

| Step | Command | Depends On |
|------|---------|------------|
| 1 | `shortlist` | Nothing |
| 1 | `suggest` | Nothing |
| 2 | `fetch-data` | Nothing (preview only — output not used by any other step) |
| 3 | `train` | Nothing (fetches its own data from yfinance) |
| 4 | `predict` | `train` — model files must exist in `data/models/` |
| 5 | `analyze` | `train` — model file must exist for the specific symbol |
| 6 | `screen` | Nothing |
| 7a | `test-buy` | Nothing (uses live price from yfinance) |
| 7a | `test-short` | Nothing (uses live price from yfinance) |
| 7b | `test-sell` | `test-buy` or `test-short` — open position must exist |
| 7c | `test-portfolio` | `test-buy` or `test-short` — open positions must exist |
| 7d | `test-calculate-gain` | `test-sell` — closed positions must exist |

---

## Output Files Summary

| Command | Console Output | CSV Output (data/reports/) | Other Files |
|---------|---------------|---------------------------|-------------|
| `shortlist` | Buy / Short / Trending tables | `shortlist.csv` | — |
| `suggest` | Ranked stocks table | `suggestions.csv` | — |
| `fetch-data` | Per-symbol row counts | — | — |
| `train` | Training progress | — | `data/models/{SYMBOL}/lstm.pt`, `xgboost.joblib`, `meta.joblib` |
| `predict` | 5 signal tables | `signals.csv`, `short_candidates.csv`, `top_picks.csv`, `sector_momentum.csv`, `news_alerts.csv` | `data/processed/predictions_YYYY-MM-DD.csv/json` (with `--export`) |
| `screen` | 4 screener tables | `top_picks.csv`, `sector_momentum.csv`, `news_alerts.csv`, `signals.csv`, `short_candidates.csv` | — |
| `analyze` | Signal + LLM analysis | `analyze.csv` (broker scores) | — |
| `test-buy` | Trade confirmation | — | `data/trades/ledger.json` |
| `test-short` | Trade confirmation | — | `data/trades/ledger.json` |
| `test-sell` | PnL confirmation | — | `data/trades/ledger.json` |
| `test-portfolio` | Open positions table | `portfolio.csv` | — |
| `test-calculate-gain` | Summary + per-stock tables | `gain_summary.csv`, `gain_per_stock.csv` | `data/trades/report_YYYY-MM-DD.json` (with `--export`) |

> All `data/reports/*.csv` files are overwritten on each run and are excluded from git.

---

## Global Options

```bash
stockpredict --log-level DEBUG <command>                  # Verbose logging
stockpredict --log-level WARNING <command>                # Quieter output
```
