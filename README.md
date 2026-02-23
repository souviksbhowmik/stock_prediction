# Indian Stock Market Prediction System

ML-powered stock prediction system for Indian markets (NSE/BSE) combining technical analysis, news sentiment, and LLM-based broker insights.

## Features

- **ML Predictions**: LSTM + XGBoost ensemble for buy/sell/hold signals
- **Technical Analysis**: RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic
- **News Sentiment**: Google News RSS + FinBERT sentiment analysis
- **LLM Broker Insights**: Ollama-powered analysis of news articles
- **Stock Suggestions**: Curated initial watchlist from NIFTY 50 ranked by momentum + news
- **Stock Screener**: Top picks, sector momentum, news alerts
- **Paper Trading**: Simulated buy/sell/short trades with gain/loss reporting

## Setup

### Create Conda Environment

```bash
conda create -n stock_prediction python=3.11 pandas numpy scikit-learn xgboost pyyaml -y
conda activate stock_prediction
pip install yfinance feedparser beautifulsoup4 requests transformers spacy ollama ta click rich joblib tqdm pytest pytest-cov ruff
pip install -e .
```

### Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Global Options

```bash
stockpredict --log-level DEBUG <command>   # Set log level (default: INFO)
```

### Fetch Data

```bash
stockpredict fetch-data --symbols RELIANCE.NS,TCS.NS
stockpredict fetch-data --symbols RELIANCE.NS --start-date 2023-01-01
stockpredict fetch-data                    # Fetches all NIFTY 50 stocks
```

### Train Models

```bash
stockpredict train --symbols RELIANCE.NS,TCS.NS
stockpredict train --symbols RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-31
stockpredict train --no-news               # Disable news features
stockpredict train --no-llm                # Disable LLM features
stockpredict train --no-news --no-llm      # Technical-only training
```

### Predict

```bash
stockpredict predict --symbols RELIANCE.NS
stockpredict predict --symbols RELIANCE.NS,TCS.NS --export   # Export CSV + JSON
stockpredict predict --no-news --no-llm                       # Technical-only predictions
```

### Analyze (Single Stock Deep Dive)

```bash
stockpredict analyze --symbol RELIANCE.NS
stockpredict analyze --symbol RELIANCE.NS --no-llm    # Skip LLM broker analysis
stockpredict analyze --symbol RELIANCE.NS --no-news   # Skip news features entirely
```

### Suggest Stocks

Get a curated watchlist of top NIFTY 50 stocks ranked by technical momentum and news sentiment.

```bash
stockpredict suggest                       # Top 10 stocks (default)
stockpredict suggest --count 5             # Top 5 stocks
stockpredict suggest --count 10 --no-news  # Technical-only ranking
```

### Stock Screener

```bash
stockpredict screen --symbols RELIANCE.NS,TCS.NS,INFY.NS
stockpredict screen                        # Screen all NIFTY 50
stockpredict screen --no-news              # Disable news-driven discovery
stockpredict screen --no-llm               # Disable LLM-based news discovery
```

### Paper Trading

Simulate trades without real money. Trades are recorded to a JSON ledger file.

```bash
# Buy Reliance worth 50,000 INR
stockpredict test-buy --symbol RELIANCE.NS --amount 50000

# Short sell TCS worth 30,000 INR
stockpredict test-short --symbol TCS.NS --amount 30000

# View open positions with unrealized PnL
stockpredict test-portfolio

# Sell Reliance (closes the long position)
stockpredict test-sell --symbol RELIANCE.NS

# Close a specific trade by ID
stockpredict test-sell --symbol TCS.NS --trade-id abc123

# Generate gain/loss report
stockpredict test-calculate-gain

# Export report to data/trades/report_YYYY-MM-DD.json
stockpredict test-calculate-gain --export
```

## Project Structure

```
├── config/
│   ├── settings.yaml          # Application settings
│   └── nifty50.yaml           # NIFTY 50 ticker list
├── src/stock_prediction/
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Configuration loader
│   ├── data/                  # Data providers (yfinance)
│   ├── features/              # Feature engineering
│   ├── models/                # ML models (LSTM, XGBoost, Ensemble)
│   ├── signals/               # Signal generation, screening, reporting, paper trading
│   ├── news/                  # RSS fetcher, sentiment, NER
│   ├── llm/                   # LLM integration (Ollama)
│   └── utils/                 # Constants, logging
├── tests/                     # Test suite
├── data/                      # Data directories (gitignored)
└── docs/                      # Documentation
```

## Running Tests

```bash
conda activate stock_prediction
pytest tests/ -v
```

## Configuration

All settings are in `config/settings.yaml` — model hyperparameters, feature parameters, signal thresholds, and paper trading paths.

## Future Enhancements

- **Generic cross-stock model**: Train a single model on a diverse set of stocks using normalized, stock-agnostic features (e.g. returns, relative volume, RSI, sector encoding) so it learns general market patterns rather than stock-specific ones. This generic model could predict any stock — including ones it was never trained on — and only needs retraining at longer intervals. The existing per-stock models would remain available for higher-accuracy predictions on frequently traded stocks.
- **Model staleness warning**: Save a `trained_at` timestamp in model metadata. When loading a model for prediction, check its age against a configurable threshold (e.g. 30 days) and display a warning if the model is stale, prompting the user to retrain.
