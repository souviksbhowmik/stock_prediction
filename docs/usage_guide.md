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

### `train` — Train Prediction Models

**What it does:**
Trains one or more ML models per symbol. When multiple models are selected they are combined into a weighted ensemble at prediction time. For each symbol the pipeline:

1. Fetches OHLCV data live from yfinance (from `--start-date` or `2020-01-01` by default)
2. Computes 30+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Stochastic, SMA/EMA crossovers, lag/momentum features)
3. Adds NIFTY 50 market context (daily return, relative strength 1d/5d)
4. Optionally fetches Google News RSS articles, runs FinBERT sentiment analysis and spaCy NER to produce 18 news features (`--no-news` to skip)
5. Optionally calls the local Ollama LLM to generate 10 broker-style insight scores per stock (`--no-llm` to skip)
6. Optionally fetches quarterly financial report data (revenue growth, margins, EPS surprise, leverage, etc.) and adds 16 point-in-time financial features (`--no-financials` to skip; enabled by default via `use_financials` in settings)
7. Creates training labels using the selected prediction horizon (default 5 trading days): at horizon H, BUY/SELL thresholds are read from `signals.horizon_thresholds` in `config/settings.yaml` (e.g. ≥2.2% → BUY, ≤-2.2% → SELL at horizon 5)
8. Builds 60-day rolling windows for sequence models (LSTM, Encoder-Decoder, Prophet) and tabular snapshots for XGBoost
9. Splits chronologically (80% train / 20% validation — no shuffling to prevent data leakage)
10. Selects best hyperparameters by **balanced accuracy** (average recall across SELL/HOLD/BUY — robust to class imbalance)
11. Retrains final model(s) on the **full dataset** using best hyperparameters
12. Saves all model artefacts to `data/models/{SYMBOL}/` and generates interactive HTML plots to `data/plots/{SYMBOL}/`

> **Note:** `train` fetches its own data directly from yfinance. Running `fetch-data` beforehand is not required.

**Model options** (`--models` / `-m`):

| Selection | Type | Description |
|---|---|---|
| `lstm` (default) | Classifier | Captures temporal patterns across 60-timestep rolling windows |
| `xgboost` | Classifier | Fast, interpretable; uses tabular feature snapshots |
| `encoder_decoder` | Regressor→signal | Seq2seq LSTM; predicts price ratios for each horizon step; Gaussian CDF → BUY/HOLD/SELL |
| `prophet` | Regressor→signal | Facebook Prophet time-series model with lag-safe exogenous regressors |
| `tft` | Regressor→signal | Temporal Fusion Transformer; gated residual networks + multi-head self-attention over sequences |
| `qlearning` | RL agent | Tabular Q-learning with discretised state space; reward = position-based P&L minus transaction cost |
| `dqn` | RL agent | Deep Q-Network; continuous-state MLP Q-function with experience replay and target network |
| `lstm,xgboost` | Ensemble | Two classifiers; combined via dynamic weights from validation balanced accuracy |
| `lstm,xgboost,encoder_decoder,prophet` | Ensemble | Four models combined; weights proportional to each model's validation balanced accuracy |
| `qlearning,dqn` | Ensemble | Both RL agents in an ensemble |

> Any combination of the seven model names works. For single-model selections the weight is 1.0. For ensembles, each model's contribution is proportional to its validation balanced accuracy — better models automatically receive higher weight.

**Prediction horizon** (`--horizon` / `-h`):

| Value | BUY threshold | SELL threshold | Description |
|---|---|---|---|
| `1` | ≥1.0% | ≤-1.0% | Next-day signal |
| `3` | ≥1.7% | ≤-1.7% | 3-day signal |
| `5` (default) | ≥2.2% | ≤-2.2% | 1-week signal |
| `7` | ≥2.6% | ≤-2.6% | 7-day signal |
| `10` | ≥3.2% | ≤-3.2% | 2-week signal |

Thresholds are configurable in `config/settings.yaml` under `signals.horizon_thresholds`. The horizon is saved in `meta.joblib` and automatically applied at prediction time.

**Dependencies:** None on prior steps. Requires internet access to yfinance and optionally Google News RSS and a running Ollama server.

**Output:**
- Console: training progress, hyperparameter search results, balanced val accuracy per symbol
- `data/reports/train_summary.csv` — per-symbol status and balanced accuracy
- Files written per symbol to `data/models/{SYMBOL}/` (only for selected models):
  - `lstm.pt` — PyTorch LSTM weights and architecture metadata
  - `xgboost.joblib` — trained XGBoost classifier with feature names
  - `encoder_decoder.pt` — PyTorch Encoder-Decoder weights
  - `prophet.joblib` — fitted Prophet model with regressors
  - `tft.pt` — PyTorch Temporal Fusion Transformer weights
  - `qlearning.joblib` — Q-table, bin edges, and state feature indices
  - `dqn.pt` — PyTorch DQN Q-network weights
  - `meta.joblib` — scalers, feature names, selected models, `trained_at` timestamp, `horizon`, `use_news`, `use_llm`, `use_financials`, `val_accuracy`, and per-model ensemble weights
- Interactive HTML plots per symbol to `data/plots/{SYMBOL}/`:
  - `train_plot.html` — training period actual vs predicted signals
  - `val_plot.html` — full history with train/val split marker
  - `pred_plot.html` — historical prices + future forecast

```bash
stockpredict train -s RELIANCE.NS,TCS.NS,SBIN.NS        # Specific stocks
stockpredict train                                        # All NIFTY 50
stockpredict train -m lstm                               # LSTM only (default)
stockpredict train -m xgboost                            # XGBoost only
stockpredict train -m encoder_decoder                    # Encoder-Decoder only
stockpredict train -m prophet                            # Prophet only
stockpredict train -m tft                                # Temporal Fusion Transformer only
stockpredict train -m qlearning                          # Tabular Q-learning only
stockpredict train -m dqn                                # Deep Q-Network only
stockpredict train -m lstm,xgboost                       # Ensemble of LSTM + XGBoost
stockpredict train -m lstm,xgboost,encoder_decoder,prophet  # Classic four-model ensemble
stockpredict train -m qlearning,dqn                      # RL-only ensemble
stockpredict train -m lstm,xgboost,tft,qlearning,dqn    # Five-model ensemble
stockpredict train -s RELIANCE.NS -h 1                   # 1-day horizon
stockpredict train -s RELIANCE.NS -h 10                  # 10-day horizon
stockpredict train -s RELIANCE.NS --no-news --no-llm     # Technical + financials only (faster)
stockpredict train -s RELIANCE.NS --no-financials        # No quarterly financial features
stockpredict train -s RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-31
```

---

## Step 4: Generate Predictions

### `predict` — Trading Signals from Trained Models

**What it does:**
Loads the trained models for each symbol and generates BUY / HOLD / SELL signals with confidence scores. For each symbol it:

1. Loads model artefacts from `data/models/{SYMBOL}/` — warns if the model is more than 30 days old
2. Reads the feature flags (`use_news`, `use_llm`, `use_financials`) and `horizon` saved in `meta.joblib` — predictions always use the exact same feature set the model was trained on
3. Fetches the latest price data from yfinance and builds current features using the stored feature set
4. Scales features using the saved StandardScaler
5. Runs prediction using the model(s) the symbol was trained with; if multiple models were trained, combines them via per-stock dynamic weights saved in `meta.joblib`
6. Applies confidence thresholds: signals below 60% confidence are downgraded to HOLD; signals above 80% are upgraded to STRONG BUY / STRONG SELL
7. Computes a short score (sell probability + RSI + MACD + SMA50) and flags short-selling candidates
8. Runs the stock screener to produce top picks, sector momentum, and news alerts alongside the model signals

> **Feature flags are not required at predict time.** The model remembers whether it was trained with news, LLM, and financial features and applies them automatically. There is no risk of feature mismatch between training and prediction.

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
```

---

## Step 5: Deep Dive (Single Stock Analysis)

### `analyze` — Detailed Single-Stock Analysis

**What it does:**
Produces a comprehensive report for a single stock combining the ML signal, technical indicator readings, LLM broker insights, and recent news headlines. It loads the trained model, generates a prediction, then sends the top 15 recent headlines to the local Ollama LLM which scores the stock across 10 dimensions (earnings outlook, competitive position, management quality, sector momentum, risk level, growth catalyst, valuation signal, institutional interest, macro impact, overall broker score) and produces a short narrative summary. LLM results are cached for 24 hours.

Feature flags are loaded from the model's `meta.joblib` automatically, matching the conditions at training time.

**Dependencies:** A trained model must exist for the symbol in `data/models/{SYMBOL}/` — run `train` first.

**Output:**
- Console: Signal summary, weekly/monthly outlook, LLM narrative summary, Broker Analysis Scores table, recent headlines
- File: `data/reports/analyze.csv` — broker score dimensions and values (written only when LLM scores are available). Overwritten on each run.

```bash
stockpredict analyze -s RELIANCE.NS                      # Full analysis
stockpredict analyze -s RELIANCE.NS --no-llm             # Skip LLM broker analysis
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

## Step 6b: Look Up a Signal by Company Name

### `lookup` — Find Signals by Name or Partial Name

**What it does:**
Searches for stocks matching any part of a company name, alias, or ticker symbol (case-insensitive substring) and returns signals for all matches. Useful when you know a company name but not its exact ticker. Three matching layers are applied:

1. Substring match in company aliases (e.g. `"sbi"` matches `"state bank of india"`)
2. Substring match in canonical company names (e.g. `"tata"` matches Tata Motors, Tata Steel, TCS, Tata Consumer)
3. Substring match in the ticker symbol itself (e.g. `"hdfc"` matches `HDFCBANK.NS`, `HDFCLIFE.NS`)

For each matched ticker it loads the trained model and generates a signal. If no model has been trained for a symbol, that row shows `N/A — No trained model` so you know which ones still need training.

**Dependencies:** Trained models in `data/models/` for signal generation. Matching itself works without any models.

**Output:**
- Console: Rich table with all matches — Symbol, Company, Signal, Confidence, BUY%, HOLD%, SELL%, Note
- File: `data/reports/lookup.csv` — same data. Overwritten on each run.

```bash
stockpredict lookup tata          # matches TCS, Tata Motors, Tata Steel, Tata Consumer
stockpredict lookup bank          # matches HDFC Bank, ICICI Bank, SBI, Axis, Kotak, IndusInd
stockpredict lookup infosys       # exact name match
stockpredict lookup "sun phar"    # partial name with space (use quotes)
stockpredict lookup hdfc          # matches by ticker prefix
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

## Streamlit UI

Launch the interactive web UI instead of the CLI:

```bash
streamlit run app.py
```

The UI mirrors all CLI functionality with additional visualisation. Key pages:

| Page | Description |
|------|-------------|
| **Train** | Select symbols, model types, horizon (1/3/5/7/10 days), and feature flags (News, LLM, Financials). Runs training in the background and streams progress. |
| **Predict** | Run predictions for trained symbols. Feature flags and horizon are loaded automatically from the model — no user input needed. Each row has an **ℹ️** button showing model metadata (trained date, horizon, val accuracy, models used, feature flags). Each row also has a **📊** button to view the interactive training plots. |
| **Analyze** | Deep-dive analysis for a single stock with LLM broker scores and chart. |
| **Suggest** | Screener-style ranked watchlist without requiring trained models. |
| **Paper Trade** | Open/close positions, view portfolio, compute gain/loss. |

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
| 6b | `lookup` | Nothing for matching; trained models in `data/models/` for signal generation |
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
| `train` | Training progress + val accuracy | `train_summary.csv` | `data/models/{SYMBOL}/lstm.pt`, `xgboost.joblib`, `encoder_decoder.pt`, `prophet.joblib`, `meta.joblib`; `data/plots/{SYMBOL}/train_plot.html`, `val_plot.html`, `pred_plot.html` |
| `predict` | 5 signal tables | `signals.csv`, `short_candidates.csv`, `top_picks.csv`, `sector_momentum.csv`, `news_alerts.csv` | `data/processed/predictions_YYYY-MM-DD.csv/json` (with `--export`) |
| `screen` | 4 screener tables | `top_picks.csv`, `sector_momentum.csv`, `news_alerts.csv`, `signals.csv`, `short_candidates.csv` | — |
| `lookup` | Matched stocks with signals | `lookup.csv` | — |
| `analyze` | Signal + LLM analysis | `analyze.csv` (broker scores) | — |
| `test-buy` | Trade confirmation | — | `data/trades/ledger.json` |
| `test-short` | Trade confirmation | — | `data/trades/ledger.json` |
| `test-sell` | PnL confirmation | — | `data/trades/ledger.json` |
| `test-portfolio` | Open positions table | `portfolio.csv` | — |
| `test-calculate-gain` | Summary + per-stock tables | `gain_summary.csv`, `gain_per_stock.csv` | `data/trades/report_YYYY-MM-DD.json` (with `--export`) |

> All `data/reports/*.csv` files are overwritten on each run and are excluded from git.

---

## meta.joblib Contents

Each trained symbol has a `data/models/{SYMBOL}/meta.joblib` file containing:

| Field | Description |
|-------|-------------|
| `scaler` | Fitted `StandardScaler` for tabular features |
| `seq_scaler` | Fitted `StandardScaler` for sequence (LSTM/ED) inputs |
| `feature_names` | Ordered list of feature column names used during training |
| `input_size` | Number of input features |
| `selected_models` | List of model names trained (e.g. `["lstm", "xgboost"]`) |
| `trained_at` | ISO timestamp of training completion |
| `horizon` | Prediction horizon in trading days (1, 3, 5, 7, or 10) |
| `use_news` | Whether news features were included during training |
| `use_llm` | Whether LLM features were included during training |
| `use_financials` | Whether quarterly financial features were included |
| `val_accuracy` | Validation balanced accuracy achieved (or `None` for old models) |
| `lstm_weight` | Ensemble weight for the LSTM model |
| `xgboost_weight` | Ensemble weight for XGBoost |
| `encoder_decoder_weight` | Ensemble weight for Encoder-Decoder |
| `prophet_weight` | Ensemble weight for Prophet |

> **Backward compatibility:** Models trained before the `horizon` / `use_*` fields were introduced will load correctly with safe defaults. Re-train to capture the new fields.

---

## Global Options

```bash
stockpredict --log-level DEBUG <command>                  # Verbose logging
stockpredict --log-level WARNING <command>                # Quieter output
```
